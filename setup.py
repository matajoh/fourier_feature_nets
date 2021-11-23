"""Setup script."""

import os
import re

import setuptools

PACKAGE_NAME = "fourier_feature_nets"
REQUIRES = [
    "scenepic",
    "numba",
    "numpy",
    "opencv-python",
    "torch",
    "matplotlib",
    "trimesh",
    "nbstripout",
    "requests",
    "progress",
    "jupyter"
]

with open("README.md", "r") as file:
    LONG_DESCRIPTION = file.read()

with open(os.path.join("fourier_feature_nets", "version.py")) as file:
    VERSION = re.search(r"\d\.\d\.\d", file.read()).group()

setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author="Matthew Johnson",
    author_email="matjoh@microsoft.com",
    description="Fourier Feature Network Lecture Resources",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    url="https://github.com/matajoh/fourier_feature_nets",
    install_requires=REQUIRES
)
