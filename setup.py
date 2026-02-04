#!/usr/bin/env python3
"""
Setup script for N4Crops package.

Installation (from n4crops directory):
    pip install -e .

With GDAL (recommended via conda):
    conda install -c conda-forge gdal
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Get the directory containing setup.py (n4crops folder)
here = Path(__file__).parent.resolve()

# Read README from parent directory (N4Maize)
readme_path = here.parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text()
else:
    # Try local README
    local_readme = here / "README.md"
    if local_readme.exists():
        long_description = local_readme.read_text()
    else:
        long_description = "N4Crops: Nitrogen Content Estimation from Hyperspectral Remote Sensing"

setup(
    name="n4crops",
    version="1.0.0",
    author="N4Maize Project",
    author_email="",
    description="Nitrogen content estimation in crops from hyperspectral imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/N4Maize",
    packages=find_packages(where="."),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "numpydoc",
        ],
    },
    entry_points={
        "console_scripts": [
            "n4crops=n4crops.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
