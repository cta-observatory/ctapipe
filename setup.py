#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# import ah_bootstrap
from setuptools import setup, find_packages
import os

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
    "ctapipe-stage1 = ctapipe.tools.stage1:main",
    "ctapipe-merge = ctapipe.tools.dl1_merge:main",
]
tests_require = ["pytest"]
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

setup(
    packages=find_packages(),
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
        "requests",
        "setuptools_scm>=3.4",
        # needed for astropy hdf5 io. Version 3 breaks copying those tables
        # with pytables du to variable length strings.
        "h5py~=2.0",
    ],
    # here are optional dependencies (as "tag" : "dependency spec")
    extras_require={
        "all": tests_require + docs_require,
        "tests": tests_require,
        "docs": docs_require,
    },
    use_scm_version={"write_to": os.path.join("ctapipe", "_version.py")},
    tests_require=tests_require,
    setup_requires=["pytest_runner", "setuptools_scm"],
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
