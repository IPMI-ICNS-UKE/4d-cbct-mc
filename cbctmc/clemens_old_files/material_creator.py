import re
import click
import numpy as np
import xcom
from scipy.integrate import simpson
from jinja2 import Environment, FileSystemLoader
from cbctmc.mc.dataio import save_text_file
import pkg_resources
import tables


def toNp(data):
    # read xcom cross-section data and return np
    x = data["energy"]
    arr = np.zeros((4, x.shape[0]))
    arr[0, :] = data["coherent"]
    arr[1, :] = data["incoherent"]
    arr[2, :] = data["photoelectric"]
    arr[3, :] = data["total"]
    return x, arr


def getAtomicMass(z):
    # use xcom data tables to return atomic mass of element with atomic number z
    z = "00" + str(z)
    with tables.open_file(xcom.NIST_XCOM_HDF5_PATH) as h5file:
        attr = h5file.get_node("/Z" + str(z[-3:]), "data").attrs
        return attr["AtomicWeight"].item()


def getCross(material, energy_max):
    # cross-section of compounds are weighted cross-sections of the involved elements, weighting factors are the mass
    # fractions of the involved elements
    compounds = parseMaterial(material)
    energy = np.arange(5000, energy_max+5, 5)
    finalcross = np.zeros((4, len(energy)))
    for com in range(len(compounds)):
        formula = compounds[com][1]
        mat = xcom.MaterialFactory.from_formula(formula)
        cross = np.zeros((4, len(energy)))
        for i in range(len(mat.elements_by_Z)):
            data = xcom.calculate_cross_section(mat.elements_by_Z[i], energy)
            en, cr = toNp(data)
            # 1/(atomic mass * 1.6605402) is factor to get from barn/atom to cm²/g (XCom returns cross-sections
            # in barn/atom)
            cross += (
                cr * mat.weights[i] / (getAtomicMass(mat.elements_by_Z[i]) * 1.6605402)
            )
        finalcross += cross * compounds[com][0]
    return finalcross


def F2(x, z):
    # calculation of Rayleigh form factor according to "Analytical cross-sections For Monte CArlo Simulation of Photon
    # Transport"; J.Baro, M.Roteta, J.M. Fernandez-Varea and F.Salvat; 1993
    par = np.loadtxt("../../tables/ParAnalyRayCrossSect")
    a = par[z - 1, 1]
    b = par[z - 1, 2]
    c = par[z - 1, 3]
    d = par[z - 1, 4]
    e = par[z - 1, 5]
    return (
        z * (1 + a * x**2 + b * x**3 + c * x**4) / (1 + d * x**2 + e * x**4) ** 2)


def F2HighEner(Q, b):
    # calculation of Rayleigh form factor according to "Analytical cross-sections For Monte CArlo Simulation of Photon
    # Transport"; J.Baro, M.Roteta, J.M. Fernandez-Varea and F.Salvat; 1993
    return np.sin(2 * b * np.arctan(Q)) / (b * Q * (1 + Q**2) ** b)


def getf2(material, energy):
    # calculation of Rayleigh form factor according to "Analytical cross-sections For Monte CArlo Simulation of Photon
    # Transport"; J.Baro, M.Roteta, J.M. Fernandez-Varea and F.Salvat; 1993
    compounds = parseMaterial(material)
    f2final = np.zeros(len(energy))
    ele_mass = 5.10998918*1e5
    # 2 * 20.6074 is factor used to change from energy to dimensionless varaible x (see Penelope eq. 2.5)
    x = (
        2 * 20.6074 * energy / ele_mass
    )  # dimensionless x(E) used for form factors

    for com in range(len(compounds)):
        formula = compounds[com][1]
        mat = xcom.MaterialFactory.from_formula(formula)
        f2 = np.zeros(len(energy))
        for i in range(len(mat.elements_by_Z)):
            z = mat.elements_by_Z[i]
            if z < 10:
                f = F2(x, z)
                f2 += f**2 * mat.weights[i] / getAtomicMass(mat.elements_by_Z[i])
            else:
                a = 1 / 137.036 * (z - 5 / 16)
                b = np.sqrt(1 - a**2)
                f = np.zeros(len(energy))
                for j in range(len(energy)):
                    e = energy[j]
                    xe = 2 * 20.6074 * e / (5.10998918 * 1e5)
                    Q = e / (a * 5.10998918 * 1e5)
                    fks = F2HighEner(Q, b)
                    fs = F2(xe, z)
                    if fs > 2:
                        f[j] = fs
                    else:
                        if fs > fks:
                            f[j] = fs
                        else:
                            f[j] = fks
                f2 += f**2 * mat.weights[i] / getAtomicMass(mat.elements_by_Z[i])
        f2final += f2 * compounds[com][0]
    return x, f2final


def getF2(formula, energy_max):
    # create propability density function by integrating and normalizing f2
    energy = np.arange(0, 300000, 5)
    energy_min = 5000
    x2, f2 = getf2(formula, energy)
    integral = np.zeros(len(f2))
    for i in range(len(f2) - 1):
        integral[i + 1] = integral[i] + simpson(f2[i : i + 2], x2[i : i + 2] ** 2)
    return integral[int(energy_min/5):int(energy_max/5+1)] / integral[-1]


def intSimf2(formula, x0, x1, norm):
    h = (x1 - x0) / 50
    x = np.arange(51) * h + x0
    e = np.sqrt(x) * (5.10998918 * 1e5) / (2 * 20.6074)
    egal, f2 = getf2(formula, e)
    f2 = f2 / norm

    inte = f2[0] * h / 3
    for i in range(24):
        inte += h / 3 * (4 * f2[(2 * i + 1)] + 2 * f2[(2 * i + 2)])
    return inte + h / 3 * (f2[50] + 4 * f2[49])


def getErr(a, b, xi, xip1, xii, xiip1, norm, formula):
    h = (xip1 - xi) / 50
    x = np.arange(51) * h + xi
    egal, f2 = getf2(formula, np.sqrt(x) * (5.10998918 * 1e5) / (2 * 20.6074))
    f2 = f2 / norm
    f = np.abs(ptilde(x, a, b, xi, xip1, xii, xiip1) - f2)

    inte = f[0] * h / 3
    for i in range(24):
        inte += h / 3 * (4 * f[(2 * i + 1)] + 2 * f[(2 * i + 2)])
    return inte + h / 3 * (f[50] + 4 * f[49])


def ritaIO(formula, energy_max):
    # rita algorithmus for fast and efficient sampling of probability density function according to
    # [Penelope 2006 Sec. 2.1.1 and Sec. 1.2.4]
    enorm = np.arange(0, 5e5, 5)
    xnorm, fnorm = getf2(formula, enorm)
    norm = simpson(fnorm, xnorm**2)
    el_mass = 5.10998918*1e5

    x2max = (energy_max / el_mass * 2 * 20.6074) ** 2
    x2 = x2max / 31 * np.arange(32)

    for j in range(128 - 32):
        print(j)
        xi = np.zeros(32 + j)
        a = np.zeros(32 + j)
        b = np.zeros(32 + j)
        err = np.zeros(31 + j)
        for i in range(len(xi) - 1):
            i += 1
            xi[i] = intSimf2(formula, x2[i - 1], x2[i], norm) + xi[i - 1]
            ei = np.array([np.sqrt(x2[i]) * (5.10998918 * 1e5) / (2 * 20.6074)])
            eim1 = np.array([np.sqrt(x2[i - 1]) * (5.10998918 * 1e5) / (2 * 20.6074)])
            dummy, f2im1 = getf2(formula, eim1)
            dummy, f2i = getf2(formula, ei)

            b[i - 1] = 1 - (
                (xi[i] - xi[i - 1]) / (x2[i] - x2[i - 1])
            ) ** 2 * norm**2 / (f2i * f2im1)
            a[i - 1] = (
                (xi[i] - xi[i - 1]) / (x2[i] - x2[i - 1]) * norm / f2im1 - b[i - 1] - 1
            )
            err[i - 1] = getErr(
                a[i - 1], b[i - 1], x2[i - 1], x2[i], xi[i - 1], xi[i], norm, formula
            )
        new = np.argmax(err)
        x2 = np.insert(x2, new + 1, (x2[new + 1] - x2[new]) / 2 + x2[new])
    xi = np.zeros(128)
    a = np.zeros(128)
    b = np.zeros(128)
    for i in range(len(xi) - 1):
        i += 1
        xi[i] = intSimf2(formula, x2[i - 1], x2[i], norm) + xi[i - 1]
        ei = np.array([np.sqrt(x2[i]) * (5.10998918 * 1e5) / (2 * 20.6074)])
        eim1 = np.array([np.sqrt(x2[i - 1]) * (5.10998918 * 1e5) / (2 * 20.6074)])
        s, f2im1 = getf2(formula, eim1)
        s, f2i = getf2(formula, ei)
        b[i - 1] = 1 - ((xi[i] - xi[i - 1]) / (x2[i] - x2[i - 1])) ** 2 * norm**2 / (
            f2i * f2im1
        )
        a[i - 1] = (
            (xi[i] - xi[i - 1]) / (x2[i] - x2[i - 1]) * norm / f2im1 - b[i - 1] - 1
        )
    return np.stack((x2, xi, a, b), axis=1)


def ptilde(x, a, b, xi, xip1, xii, xiip1):
    # rita algorithmus for fast and efficient sampling of probability density function accroing to
    # [Penelope 2006 Sec. 2.1.1 and Sec. 1.2.4]
    tau = (x - xi) / (xip1 - xi)
    tau[0] = 0.5
    c = 1 - (4 * b * tau**2) / (1 + a + b - a * tau) ** 2
    c = np.abs(c)
    nu = (1 + a + b - a * tau) / (2 * b * tau) * (1 - np.sqrt(c))
    nu[0] = 0
    1 / (1 - b * nu**2)
    return (
        (1 + a * nu + b * nu**2) ** 2
        / ((1 + a + b) * (1 - b * nu**2))
        * (xiip1 - xii)
        / (xip1 - xi)
    )


def limitsBinSearch(p):
    # rita algorithmus for fast and efficient sampling of probability density function according to
    # [Penelope 2006 Sec. 2.1.1 and Sec. 1.2.4]
    no = len(p)
    uplim = np.zeros(no)
    lolim = np.zeros(no)
    for i in range(no):
        for j in range(no):
            j = 127 - j
            if i / (no - 1) >= p[j]:
                lolim[i] = j + 1
                break
        for j in range(no):
            if (i + 1) / (no - 1) <= p[j]:
                uplim[i] = j + 1
                break
    uplim[-1] = no
    lolim[-1] = 1
    return lolim, uplim


def parseMaterial(material):
    # split material input into molecules and corresponding mass fraction
    compounds = material.split("_")
    for i in range(len(compounds)):
        data = []
        compounds[i] = compounds[i].split(":")
        if len(compounds[i]) == 1:
            compounds[i] = [1, compounds[i][0]]
        compounds[i][0] = float(compounds[i][0])
    return compounds


def getJ0(material):
    # see [Penelope 2006, Sec. 2.3, especially 2.3.1], tables/compton is taken from:
    #      [F.Biggs, L.B.Mendelsohn, J.B.Mann 1975, Hartree Fock compten profiles for the elements]
    compounds = parseMaterial(material)
    compten = np.genfromtxt(
        "../../tables/compten", skip_header=1, usecols=range(19), delimiter="\t"
    )
    data = []
    alpha = 1 / (137.036)
    for i in range(len(compounds)):
        comp = xcom.MaterialFactory.from_formula(compounds[i][1])
        elementsbyz = comp.elements_by_Z
        weights = comp.weights
        noatoms = getNoAtoms(compounds[i][1])
        for j in range(len(elementsbyz)):
            rowcomp = compten[elementsbyz[j] - 1]
            for k in range(int(len(rowcomp) / 3)):
                k = k * 3 + 1
                if not np.isnan(rowcomp[k]):
                    data.append(
                        [
                            rowcomp[k + 1] * noatoms[j] * compounds[i][0],
                            rowcomp[k + 2],
                            rowcomp[k] * 1 / alpha,
                            elementsbyz[j],
                            0,
                        ]
                    )
    return np.array(data)


def getNoAtoms(formula):
    ele = re.findall(r"([A-Z][a-z]*)([1-9])*", formula)
    for i in range(len(ele)):
        mat = xcom.MaterialFactory.from_formula(ele[i][0])
        if ele[i][1] == "":
            ele[i] = [mat.elements_by_Z[0], 1]
        else:
            ele[i] = [mat.elements_by_Z[0], int(ele[i][1])]
    ele = np.array(ele)
    ele = ele[ele[:, 0].argsort()]
    return ele[:, 1]

def convert_numpy_to_string(x):
    string = ""
    for row in x:
        string += (' '.join(map(lambda x: "{}".format(x), row))) + "\n"
    return string[:-2]

# @click.command()
# @click.option(
#     "--formula",
#     help="Enter chemical material definition. For every compound give mass fraction "
#     "and chemical formula seperated with : every compound is seperated with _."
#     'Examples: "H2O", "0.78:N_0.21:O_0.01:Ar", "0.965:H2O_0.035:NaCl"',
# )
# @click.option("--name", help="Material name")
# @click.option("--path", help="Path for Output")
# @click.option("--roh", help="Density in g/cm³")
# @click.option("--energy_max", default=125000, type=int, help="Insert the maximal röntgen spectra energy in eV. example:"
#                                                              " if the CBCT has a 125kV röntgen source, insert 125000")
def createMat(formula, name, path, roh, energy_max):
    # Writes Material File for MC-GPU use, energy range is from 5keV to energy_max in 5eV steps
    energy_min = 5000
    energy = np.arange(energy_min, energy_max+5, 5)
    cross = getCross(material=formula, energy_max=energy_max)
    roh = float(roh)
    mFreePath = (cross * roh) ** -1
    # concatenate Energy spektrum, mean free path and rayleigh form factor for the final output data
    freepath = np.concatenate(
        (
            np.concatenate(
                (energy.reshape((int((energy_max - energy_min)/5)+1, 1)), np.swapaxes(mFreePath, 0, 1)), axis=1
            ),
            getF2(formula, energy_max).reshape((int((energy_max - energy_min)/5)+1, 1)),
        ),
        axis=1,
    )
    # the rita algorithm is explained in [Penelope 2006 Sec. 2.1.1 and Sec. 1.2.4]
    rita = ritaIO(formula, energy_max)
    lim = np.stack(limitsBinSearch(rita[:, 1]))
    lim = np.swapaxes(lim, 0, 1)
    rita = np.concatenate((rita, lim), axis=1)

    compton = getJ0(formula)
    compton_len = str(len(compton))
    compton = compton[compton[:, 1].argsort()]
    params = {
        "name": name + "(" + formula + ")",
        "roh": roh,
        "freepath": convert_numpy_to_string(freepath),
        "rita": convert_numpy_to_string(rita),
        "compton_len": compton_len,
        "compton_data": convert_numpy_to_string(compton),
    }
    assets_folder = pkg_resources.resource_filename("cbctmc", "assets/templates")
    environment = Environment(loader=FileSystemLoader(assets_folder))
    template = environment.get_template("mcgpu_material.jinja2")
    rendered = template.render(params)

    save_text_file(rendered, path+"/"+name+".mcgpu", compress=False)


if __name__ == "__main__":
    createMat("0.9:H2O_0.1:NaCl", "salzwasser_nachher",  "/home/crohling/Documents/test_mat", 1.0, 125000)

