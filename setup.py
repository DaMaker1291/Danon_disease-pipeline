from setuptools import setup, find_packages

setup(
    name="longevity-pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "fair-esm>=2.0.0",
        "rdkit>=2023.9.0",
        "biopython>=1.81",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.1.0",
        "ray[default]>=2.9.0",
        "mpi4py>=3.1.0",
        "pydantic>=2.5.0",
        "h5py>=3.10.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "longevity-pipeline=pipeline.main:main",
        ],
    },
)
