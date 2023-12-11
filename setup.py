from setuptools import find_packages, setup

setup(
    name="cbctmc",
    version="0.1",
    author="UKE ICNS IPMI Group",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "run-mc = scripts.run_mc_simulations:run",
            "run-mc-lp = scripts.run_mc_line_pairs:run",
            "recon-mc = cbctmc.reconstruction.reconstruction:_cli",
            "fit-noise = scripts.fit_noise:cli",
        ],
    },
    package_data={"cbctmc": ["*.jinja2", "*.mcgpu"]},
)
