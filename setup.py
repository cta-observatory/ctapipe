#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# import ah_bootstrap
from setuptools import setup, find_packages

import sys
import os

# pep 517 builds do not have cwd in PATH by default
sys.path.insert(0, os.path.dirname(__file__))
# Get the long and version description from the package's docstring
import ctapipe  # noqa


# Define entry points for command-line scripts
# TODO: this shuold be automated (e.g. look for main functions and
# rename _ to -, and prepend 'ctapipe'
entry_points = {}
entry_points["console_scripts"] = [
    "ctapipe-info = ctapipe.tools.info:main",
    "ctapipe-camdemo = ctapipe.tools.camdemo:main",
    "ctapipe-dump-triggers = ctapipe.tools.dump_triggers:main",
    "ctapipe-chargeres-extract = ctapipe.tools.extract_charge_resolution:main",
    "ctapipe-chargeres-plot = ctapipe.tools.plot_charge_resolution:main",
    "ctapipe-dump-instrument=ctapipe.tools.dump_instrument:main",
    "ctapipe-event-viewer = ctapipe.tools.bokeh.file_viewer:main",
    "ctapipe-display-tel-events = ctapipe.tools.display_events_single_tel:main",
    "ctapipe-display-imagesums = ctapipe.tools.display_summed_images:main",
    "ctapipe-reconstruct-muons = ctapipe.tools.muon_reconstruction:main",
    "ctapipe-display-integration = ctapipe.tools.display_integrator:main",
    "ctapipe-display-dl1 = ctapipe.tools.display_dl1:main",
    "ctapipe-stage1-process = ctapipe.tools.stage1:main",
]
tests_require = [
    "pytest",
    "ctapipe-extra @ https://github.com/cta-observatory/ctapipe-extra/archive/v0.3.0.tar.gz",
]
docs_require = [
    "sphinx_rtd_theme",
    "sphinx_automodapi",
    "sphinx",
    "nbsphinx",
    "numpydoc",
    "jupyter",
    "notebook",
    "travis-sphinx",
    "graphviz",
]

ctapipe.version.update_release_version()

setup(
    packages=find_packages(),
    version=ctapipe.version.get_version(pep440=True),
    python_requires=">=3.7",
    install_requires=[
        "astropy>=3,<5",
        "bokeh~=1.0",
        "eventio>=1.1.1,<2.0.0a0",  # at least 1.1.1, but not 2
        "iminuit>=1.3",
        "joblib",
        "matplotlib~=3.0",
        "numba>=0.43",
        "numpy~=1.16",
        "pandas>=0.24.0",
        "psutil",
        "scikit-learn",
        "scipy~=1.2",
        "tables~=3.4",
        "tqdm>=4.32",
        "traitlets~=5.0,>=5.0.5",
        "zstandard",
        "h5py",  # needed for astropy hdf5 io
    ],
    # here are optional dependencies (as "tag" : "dependency spec")
    extras_require={
        "all": tests_require + docs_require,
        "tests": tests_require,
        "docs": docs_require,
    },
    tests_require=tests_require,
    setup_requires=["pytest_runner"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Development Status :: 3 - Alpha",
    ],
    zip_safe=False,
    entry_points=entry_points,
    package_data={"": ["tools/bokeh/*.yaml", "tools/bokeh/templates/*.html"]},
)
