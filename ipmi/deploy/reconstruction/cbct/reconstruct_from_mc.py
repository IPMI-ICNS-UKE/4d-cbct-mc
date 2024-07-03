import logging
from pathlib import Path
from typing import Optional

import click
import yaml

from ipmi.common.decorators import convert
from ipmi.common.logger import init_fancy_logging
from ipmi.reconstruction.cbct.reconstructors import FDKReconstructor

logger = logging.getLogger(__name__)
init_fancy_logging()

logging.getLogger(__name__).setLevel(logging.DEBUG)
logging.getLogger("ipmi").setLevel(logging.DEBUG)


@convert("projections_filepath", converter=Path)
@convert("geometry_filepath", converter=Path)
@convert("output_folder", converter=Path)
def reconstruct(
    projections_filepath: Path,
    geometry_filepath: Path,
    output_folder: Optional[Path] = None,
    method: str = "fdk3d",
    **kwargs,
):
    if not output_folder:
        # name of zip file as folder
        output_folder = projections_filepath.parent / "reconstruction"

    output_folder.mkdir(parents=True, exist_ok=True)

    if method == "fdk3d":
        reconstructor = FDKReconstructor(use_docker=True)
        fdk_params = dict(
            path=projections_filepath.parent,
            regexp=projections_filepath.name,
            geometry=geometry_filepath,
            hardware="cuda",
            pad=kwargs["pad"],
            hann=kwargs["hann"],
            hannY=kwargs["hann_y"],
            dimension=kwargs["dimension"],
            spacing=kwargs["spacing"],
            output_filepath=output_folder / "3d_fdk.mha",
        )
        reconstructor.reconstruct(**fdk_params)

        with open(output_folder / "3d_fdk_params.yaml", "w") as f:
            yaml.dump(fdk_params, f)
    else:
        raise NotImplementedError(method)


@click.group()
def cli():
    pass


@click.command()
@click.argument(
    "projections-filepath",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "geometry-filepath",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--dimension",
    type=click.Tuple([int, int, int]),
    default=(464, 250, 464),
    help="Image dimension of the reconstruction",
    show_default=True,
)
@click.option(
    "--spacing",
    type=click.Tuple([float, float, float]),
    default=(1.0, 1.0, 1.0),
    help="Image spacing of the reconstruction",
    show_default=True,
)
@click.option(
    "--pad",
    type=click.FLOAT,
    default=1.0,
    help="Data padding parameter to correct for truncation",
    show_default=True,
)
@click.option(
    "--hann",
    type=click.FLOAT,
    default=1.0,
    help="Cut frequency for hann window in ]0,1] (0.0 disables it)",
    show_default=True,
)
@click.option(
    "--hann-y",
    type=click.FLOAT,
    default=1.0,
    help="Cut frequency for hann window in ]0,1] (0.0 disables it)",
    show_default=True,
)
def fdk3d(*args, **kwargs):
    reconstruct(*args, method="fdk3d", **kwargs)


cli.add_command(fdk3d)


if __name__ == "__main__":
    cli()

    # cli([
    #     "fdk3d",
    #     "/media/fmadesta/USB/Nov_15/sim_bin_03.mha",
    #     "/media/fmadesta/USB/Nov_15/bin_03.xml"
    # ])

    # cli(
    #     [
    #         "rooster4d",
    #         "/datalake/recon_laura/306/session1/2020-07-13_093733.zip",
    #         "--output-folder",
    #         "/datalake/recon_laura/306/session1/recon_fm",
    #         "--binning",
    #         "phase_varian",
    #         "--plot",
    #     ]
    # )

    # cli(
    #     [
    #         "rooster4d",
    #         "/datalake/recon_laura/319/session1/2020-09-28_094710.zip",
    #         "--output-folder",
    #         "/datalake/recon_laura/319/session1/recon_fm",
    #         "--binning",
    #         "phase_varian",
    #         "--plot"
    #     ]
    # )
