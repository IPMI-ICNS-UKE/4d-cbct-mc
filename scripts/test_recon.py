from pathlib import Path

from cbctmc.reconstruction.reconstruction import reconstruct_3d

reconstruct_3d(
    projections_filepath=Path(
        "/datalake_fast/mc_test/mc_output/geometry_test/run_2023-07-04T15:23:36.970465/projections_total_normalized.mha"
    ),
    geometry_filepath=Path(
        "/datalake_fast/mc_test/mc_output/geometry_test/geometry.xml"
    ),
    output_folder=Path(
        "/datalake_fast/mc_test/mc_output/geometry_test/run_2023-07-04T15:23:36.970465/recon"
    ),
)

reconstruct_3d(
    projections_filepath=Path(
        "/datalake_fast/mc_test/mc_output/geometry_test/run_2023-07-04T15:23:36.970465/projections_total_normalized.mha"
    ),
    geometry_filepath=Path(
        "/datalake_fast/mc_test/mc_output/geometry_test/geometry.xml"
    ),
    output_folder=Path(
        "/datalake_fast/mc_test/mc_output/geometry_test/run_2023-07-04T15:23:36.970465/recon"
    ),
)
