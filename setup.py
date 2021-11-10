"""Setup script."""

import os
import re

import setuptools

PACKAGE_NAME = "nerf_lecture"
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

with open(os.path.join("nerf", "version.py")) as file:
    VERSION = re.search(r"\d\.\d\.\d", file.read()).group()

setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author="Matthew Johnson",
    author_email="matjoh@microsoft.com",
    description="NeRF Lecture Resources",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    url="https://github.com/matajoh/nerf_lecture",
    install_requires=REQUIRES
)
