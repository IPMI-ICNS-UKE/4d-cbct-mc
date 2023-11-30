from __future__ import annotations

import os

import click
import numpy as np
import pkg_resources
import xraydb
from jinja2 import Environment, FileSystemLoader
from scipy import integrate
from tqdm import tqdm

from cbctmc.mc.dataio import save_text_file


class MaterialData:
    def __init__(
        self, name: str, formula: str, density: float, energy_max: int, path_saving: str
    ):
        self._name = name
        self._formula = formula
        self._density = density
        self._energy_max = energy_max
        self._energy_min = 5000
        self._energy_spectrum = np.arange(self._energy_min, self._energy_max + 5, 5)
        self._path_saving = path_saving

    def createMaterialDataFile(self):
        # this function creates a material file which MC-GPU can use to simulate photon transport through that material

        # first part of the output data contains the energy spectrum, the mean free paths and the rayleigh form factor
        # as a cumulative density function (cdf)
        meanFreePath = self.getMeanFreePath()

        x, form_factor_squared = self.getFormFactorSquared(
            np.arange(0, 2 * self._energy_max + 5, 5)
        )
        # the form factor is normalized to create a probability density function (pdf) and then integrated over x² from
        # 0 to 2*the maximal Energy (since (2*max E)/c ist the maximal momentum transfer) to create the cdf,
        # then the entries according to self._energy_spectrum are selected

        pdf = form_factor_squared / integrate.simpson(form_factor_squared, x**2)
        cdf = integrate.cumtrapz(
            pdf[: int((self._energy_max + 5) / 5)],
            x[: int((self._energy_max + 5) / 5)] ** 2,
            initial=0,
        )
        cdf = cdf[int(self._energy_min / 5) :]

        # reshape data to create n rows with 6 entries, the entries are :
        # [Energy (eV) | Rayleigh | Compton | Photoelectric | TOTAL (+pair prod) (cm) | Rayleigh: max cumul prob F^2]
        first_data_part = np.concatenate(
            (
                np.concatenate(
                    (
                        self._energy_spectrum.reshape(
                            (int((self._energy_max - self._energy_min) / 5) + 1, 1)
                        ),
                        np.swapaxes(meanFreePath, 0, 1),
                    ),
                    axis=1,
                ),
                cdf.reshape((int((self._energy_max - self._energy_min) / 5) + 1, 1)),
            ),
            axis=1,
        )

        # the second data part uses the RITA Algorithm [Penelope 2006 Sec. 2.1.1 and Sec. 1.2.4] to create 128
        # datapoints containing the form factor squared as a cdf of x², and for each datapoint the parameters a and b
        # which are used to interpolate between two data points
        x_squared, cdf, a, b = self.ritaIO()
        # for each datapoint the upper and lower limit are introduced, so that MC-GPU can efficiently find the
        # correct interval for a sampled random value between 0 and 1
        lower_limit, upper_limit = MaterialData.limitsBinSearch(cdf)
        # group second data part in the form:
        # [x² | CDF | a | b | lower_limit | upper_limit]
        second_data_part = np.stack([x_squared, cdf, a, b, lower_limit, upper_limit])
        second_data_part = second_data_part.swapaxes(0, 1)
        # the third data_part contains electron shell information of the involved elements and is based on
        # [Penelope 2006, Sec. 2.3, especially 2.3.1], tables/compton is taken from:
        # [F.Biggs, L.B.Mendelsohn, J.B.Mann 1975, Hartree Fock compton profiles for the elements]
        third_data_part = self.getShellInformation()

        # use the template to create the Data file

        params = {
            "name": self._name + "(" + self._formula + ")",
            "roh": self._density,
            "freepath": MaterialData.convert_numpy_to_string(first_data_part),
            "rita": MaterialData.convert_numpy_to_string(second_data_part),
            "compton_len": str(len(third_data_part)),
            "compton_data": MaterialData.convert_numpy_to_string(third_data_part),
        }
        assets_folder = pkg_resources.resource_filename("cbctmc", "assets/templates")
        environment = Environment(loader=FileSystemLoader(assets_folder))
        template = environment.get_template("mcgpu_material.jinja2")
        rendered = template.render(params)
        if not os.path.exists(self._path_saving):
            os.makedirs(self._path_saving)
        save_text_file(
            rendered, self._path_saving + "/" + self._name + ".mcgpu", compress=False
        )

    def getMeanFreePath(self):
        # this function returns the mean free paths for rayleigh, compton, photoelectric and total scattering for the
        # specified compound
        # for every molecule in the compound, the molecular attenuation is calculated out of the atomic attenuation
        # values given by th databanks of xraydb. Each atomic attenuation is weighted by the mass fraction of the
        # element in the molecule
        # total attenuation is calculated out of the sum of the molecular attenuation weigted with the mass fraction of
        # each molecule, given by the user input

        energy = self._energy_spectrum
        mass_attenuation = np.zeros((4, len((energy))))

        molecules = self.parseCompoundToMolecules(self._formula)
        for molecule in molecules:
            elements = xraydb.chemparse(molecule)
            molecular_mass = sum(
                xraydb.atomic_mass(element) * elements[element] for element in elements
            )
            mass_attenuation_molecule = np.zeros((4, len((energy))))
            for element in elements:
                molecular_mass_fraction = (
                    xraydb.atomic_mass(element) * elements[element] / molecular_mass
                )
                mass_attenuation_molecule[0] += (
                    xraydb.mu_elam(element, energy, kind="coh")
                    * molecular_mass_fraction
                )
                mass_attenuation_molecule[1] += (
                    xraydb.mu_elam(element, energy, kind="incoh")
                    * molecular_mass_fraction
                )
                mass_attenuation_molecule[2] += (
                    xraydb.mu_elam(element, energy, kind="photo")
                    * molecular_mass_fraction
                )
                mass_attenuation_molecule[3] += (
                    xraydb.mu_elam(element, energy, kind="total")
                    * molecular_mass_fraction
                )
            mass_attenuation += mass_attenuation_molecule * molecules[molecule]
        mean_free_path = (mass_attenuation * self._density) ** -1
        return mean_free_path

    def getFormFactorSquared(self, energy_range: np.array):
        # this function returns the form factor of the material squared
        # the form factor is calculated  according to "Analytical cross-sections For Monte CArlo Simulation of
        # Photon Transport"; J.Baro, M.Roteta, J.M. Fernandez-Varea and F.Salvat; 1993
        # a step by step explanation of the calculation can be found in "PENELOPE-2006: A Code System
        # for Monte Carlo Simulation of Electron and Photon Transport";Francesc Salvat, José M. Fernández-Varea,
        # Josep Sempau; 2006; Sec. 2.1

        energy = energy_range
        form_factor_squared = np.zeros(len(energy))
        electron_mass = 5.10998918 * 1e5
        conversion_factor = 2 * 20.6074
        # 2 * 20.6074 is factor used to convert energy to dimensionless variable x (see Penelope eq. 2.5)
        x = conversion_factor * energy / electron_mass
        # iterate over all molecules in the material
        molecules = self.parseCompoundToMolecules(self._formula)
        for molecule in molecules:
            form_factor_squared_molecule = np.zeros(len(energy))
            # iterate over the elements in the molecule
            elements = xraydb.chemparse(molecule)
            molecular_mass = sum(
                xraydb.atomic_mass(element) * elements[element] for element in elements
            )
            for element in elements:
                z = xraydb.atomic_number(element)
                molecular_mass_fraction = (
                    xraydb.atomic_mass(element) * elements[element] / molecular_mass
                )
                # the atomic form factor for elements is either calculated with an approximation which parameters have
                # been fitted to experimental data or with a theoretical approximation
                # the fitted parameters are listed in "Analytical cross-sections For Monte Carlo Simulation of Photon
                # Transport"; J.Baro, M.Roteta, J.M. Fernandez-Varea and F.Salvat; 1993
                par = np.loadtxt("../../tables/ParAnalyRayCrossSect")
                par = par[z - 1, 1:]
                fitted_atomic_form_factor = (
                    z
                    * (1 + par[0] * x**2 + par[1] * x**3 + par[2] * x**4)
                    / ((1 + par[3] * x**2 + par[4] * x**4) ** 2)
                )
                if z < 10:
                    form_factor_squared_molecule += (
                        fitted_atomic_form_factor**2 * molecular_mass_fraction
                    )
                else:
                    atomic_form_factor = np.zeros(len(energy))
                    for j in range(len(energy)):
                        if fitted_atomic_form_factor[j] > 2:
                            atomic_form_factor[j] = fitted_atomic_form_factor[j]
                        else:
                            if fitted_atomic_form_factor[
                                j
                            ] > MaterialData.theoreticalFormFactor(energy[j], z):
                                atomic_form_factor[j] = fitted_atomic_form_factor[j]
                            else:
                                atomic_form_factor[
                                    j
                                ] = MaterialData.theoreticalFormFactor(energy[j], z)
                    form_factor_squared_molecule += (
                        atomic_form_factor**2 * molecular_mass_fraction
                    )
            form_factor_squared += form_factor_squared_molecule * molecules[molecule]
        return x, form_factor_squared

    def ritaIO(self):
        # rita algorithm for fast and efficient sampling of the probability density function (normalized form factor
        # squared) according to [Penelope 2006 Sec. 1.2.4]
        # the form factor is used to sample the change of photon direction due to Rayleigh scattering
        # to sample, the inverse-transform method is used, for which the cumulative density function (cdf) of the
        # underlying probability density function (pdf) has to be calculated

        # first the form factor has to be normalized to gain a pdf
        # we normalize the form factor by integrating over all possible momentum transfer (q=0 to q_max=2*e_max/c)
        # since in MC-GPU x² is sampled, we will be integrating over x² (x depends on e: x_max = x(e_max))

        energy_range_normalization = np.arange(0, int(2 * self._energy_max) + 1, 1)
        x_range_normalization, form_factor_squared = self.getFormFactorSquared(
            energy_range_normalization
        )
        x_squared = x_range_normalization**2
        norm = integrate.simpson(form_factor_squared, x_squared)

        pdf = form_factor_squared / norm
        cdf = integrate.cumtrapz(pdf, x_range_normalization**2, initial=0)

        # initially 32 datapoints evenly spaced in the x²_range are taken, with the factors a and b one can interpolate
        # between these datapoints
        # the mean error between the actual form factor squared and the interpolated result is calculated for each of
        # the 31 intervals, a new datapoint is added in the middle of the interval with the highest mean error
        # the process is repeated until 128 datapoints are selected, for each datapoint the factors a and b are
        # calculated
        sample_indices = np.arange(
            0, int(len(x_squared) / 32) * 31, int(len(x_squared) / 32)
        )
        sample_indices = np.append(sample_indices, len(x_squared) - 1)

        # 96 grid points are added in the middle of the intervals which have the biggest error in that iteration

        for i in tqdm(range(97)):
            # b is taken from [Penelope 2006 Eq. 1.53b]
            b = 1 - (
                (
                    (cdf[sample_indices[1:]] - cdf[sample_indices[:-1]])
                    / (x_squared[sample_indices[1:]] - x_squared[sample_indices[:-1]])
                )
                ** 2
            ) * 1 / (pdf[sample_indices[1:]] * pdf[sample_indices[:-1]])
            # a is taken from [Penelope 2006 Eq. 1.53a]
            a = (
                (
                    (cdf[sample_indices[1:]] - cdf[sample_indices[:-1]])
                    / (x_squared[sample_indices[1:]] - x_squared[sample_indices[:-1]])
                )
                / pdf[sample_indices[:-1]]
                - b
                - 1
            )
            # the mean error between the interpolation and the original pdf is calculated for each interval, in the
            # first iteration the mean error of the 31 initial intervals are calculated
            if i == 0:
                error = np.array(
                    [
                        integrate.simpson(
                            np.abs(
                                pdf[l:u]
                                - MaterialData.rationalInterpolation(
                                    x_squared[l:u],
                                    x_squared[sample_indices],
                                    cdf[sample_indices],
                                    a,
                                    b,
                                )
                            ),
                            x_squared[l:u],
                        )
                        for l, u in zip(sample_indices[:-1], sample_indices[1:])
                    ]
                )
            # in all other iteration the error of the  new intervals are calculated,
            # in the last iteration the final a and b are calculated, a new error is not necessary
            elif i < 96:
                error_new = np.array(
                    [
                        integrate.simpson(
                            np.abs(
                                pdf[l:u]
                                - MaterialData.rationalInterpolation(
                                    x_squared[l:u],
                                    x_squared[sample_indices],
                                    cdf[sample_indices],
                                    a,
                                    b,
                                )
                            ),
                            x_squared[l:u],
                        )
                        for l, u in zip(
                            sample_indices[max_error_interval : max_error_interval + 2],
                            sample_indices[
                                max_error_interval + 1 : max_error_interval + 3
                            ],
                        )
                    ]
                )
                error[max_error_interval] = error_new[0]
                error = np.insert(error, max_error_interval + 1, error_new[1])
            if i < 96:
                max_error_interval = np.argmax(error)
                # calculate new grid point in the middle of the interval and corresponding factors a and b
                new_grid_point = (
                    sample_indices[max_error_interval + 1]
                    + sample_indices[max_error_interval]
                ) / 2
                sample_indices = np.insert(
                    sample_indices, max_error_interval + 1, int(new_grid_point)
                )
        a = np.append(
            a, 0
        )  # add last entry 0 to have the same number of entries as data grid points
        b = np.append(b, 0)
        return x_squared[sample_indices], cdf[sample_indices], a, b

    def getShellInformation(self):
        # see [Penelope 2006, Sec. 2.3, especially 2.3.1], tables/compton is taken from:
        # [F.Biggs, L.B.Mendelsohn, J.B.Mann 1975, Hartree Fock compten profiles for the elements]
        molecules = self.parseCompoundToMolecules(self._formula)
        compten = np.genfromtxt(
            "../../tables/compten", skip_header=1, usecols=range(19), delimiter="\t"
        )
        shell_information = []
        alpha = 1 / (137.036)
        for molecule in molecules:
            elements = xraydb.chemparse(molecule)
            # noatoms = self.getNoAtoms(molecules[i][1])
            for element in elements:
                noatoms = elements[element]
                element_z = xraydb.atomic_number(element)
                element_compten_data = compten[element_z - 1]
                for k in range(int(len(element_compten_data) / 3)):
                    k = k * 3 + 1
                    if not np.isnan(element_compten_data[k]):
                        shell_information.append(
                            [
                                element_compten_data[k + 1]
                                * noatoms
                                * molecules[molecule],
                                element_compten_data[k + 2],
                                element_compten_data[k] * 1 / alpha,
                                element_z,
                                0,
                            ]
                        )
        shell_information = np.array(shell_information)
        shell_information = shell_information[shell_information[:, 1].argsort()]
        return shell_information

    @staticmethod
    def theoreticalFormFactor(energy: np.array, z: int):
        # "PENELOPE-2006: A Code System for Monte Carlo Simulation of Electron and Photon Transport";Francesc Salvat,
        # José M. Fernández-Varea, Josep Sempau; 2006; Equations 2.8 and 2.9
        fine_structure_constant = 1 / 137.036
        electron_mass = 5.10998918 * 1e5
        a = fine_structure_constant * (z - 5 / 16)
        b = np.sqrt(1 - a**2)
        Q = energy / (a * electron_mass)
        return np.sin(2 * b * np.arctan(Q)) / (b * Q * (1 + Q**2) ** b)

    @staticmethod
    def limitsBinSearch(cdf):
        length = len(cdf)
        uplim = np.zeros(length)
        lolim = np.zeros(length)
        for i in range(length):
            for j in range(length):
                j = 127 - j
                if i / (length - 1) >= cdf[j]:
                    lolim[i] = j + 1
                    break
            for j in range(length):
                if (i + 1) / (length - 1) <= cdf[j]:
                    uplim[i] = j + 1
                    break
        uplim[-1] = length
        lolim[-1] = 1
        return lolim.astype(int), uplim.astype(int)

    @staticmethod
    def rationalInterpolation(x, x_samples, cdf_samples, a, b):
        # data samples of PDF(x) are interpolated according to [Penelope 2006 Eq. 1.55 and 1.56]
        pdf_interpolated = np.zeros(len(x))
        for j in range(len(x)):
            for i in range(len(x_samples) - 1):
                if x_samples[i] <= x[j] < x_samples[i + 1]:
                    tau = (x[j] - x_samples[i]) / (x_samples[i + 1] - x_samples[i])
                    if x[j] == x_samples[i]:
                        nu = 0
                    else:
                        nu = ((1 + a[i] + b[i] - a[i] * tau) / (2 * b[i] * tau)) * (
                            1
                            - np.sqrt(
                                1
                                - 4
                                * b[i]
                                * tau**2
                                / (1 + a[i] + b[i] - a[i] * tau) ** 2
                            )
                        )
                    pdf_interpolated[j] = (
                        (1 + a[i] * nu + b[i] * nu**2) ** 2
                        * (cdf_samples[i + 1] - cdf_samples[i])
                        / (
                            (1 + a[i] + b[i])
                            * (1 - b[i] * nu**2)
                            * (x_samples[i + 1] - x_samples[i])
                        )
                    )
                    break
        return pdf_interpolated

    @staticmethod
    def parseCompoundToMolecules(formula: str):
        # split compund formula input into molecules and corresponding mass fractions
        if len(formula.split(":")) == 1:
            return {formula: 1}
        molecules = dict(
            (molecule.strip(), float(mass_fraction.strip()))
            for mass_fraction, molecule in (
                group.split(":") for group in formula.split("_")
            )
        )
        return molecules

    @staticmethod
    def convert_numpy_to_string(x):
        string = ""
        for row in x:
            string += (" ".join(map(lambda x: "{}".format(x), row))) + "\n"
        return string[:-2]


@click.command()
@click.option("--name", help="Material name")
@click.option(
    "--formula",
    help="Enter chemical material definition. For every compound list all molecules and mass fraction of that molecule"
    "in the compound, in the form: mass_fraction1:molecule1_mass_fraction2:molecule2. "
    '"Examples: "H2O", "0.78:N_0.21:O_0.01:Ar", "0.965:H2O_0.035:NaCl"',
)
@click.option("--density", help="Density in g/cm³", type=float)
@click.option(
    "--energy_max",
    default=125000,
    type=int,
    help="Insert the maximal röntgen spectra energy in eV. "
    "example: if the CBCT has a 125kV röntgen source, "
    "insert 125000",
)
@click.option("--path_saving", help="Path for Output")
def createMaterialDataObject(
    name: str, formula: str, density: float, energy_max: int, path_saving: str
):
    material_obj = MaterialData(
        name=name,
        formula=formula,
        density=density,
        energy_max=energy_max,
        path_saving=path_saving,
    )
    material_obj.createMaterialDataFile()
    return material_obj


if __name__ == "__main__":
    createMaterialDataObject()
