# write the setup script

from setuptools import setup, find_packages


setup(
    name="mace_fep",
    version="0.1",
    description="MACE FEP",
    author="Harry Moore",
    packages=find_packages(),
    install_requires=[],
    entry_points={
        "console_scripts": [
            "mace-fep=mace_fep.entrypoint:main",
        ],
    },
)
