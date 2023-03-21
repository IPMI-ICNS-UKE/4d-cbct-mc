from setuptools import find_packages, setup

setup(
    name="cbctmc",
    version="0.1",
    author="UKE ICNS IPMI Group",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "run_mc = cbctmc.main:run",
        ],
    },
    package_data={"cbctmc": ["*.jinja2", "*.mcgpu"]},
)
