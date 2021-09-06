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
    "ctapipe-dump-instrument=ctapipe.tools.dump_instrument:main",
    "ctapipe-event-viewer = ctapipe.tools.bokeh.file_viewer:main",
    "ctapipe-display-tel-events = ctapipe.tools.display_events_single_tel:main",
    "ctapipe-display-imagesums = ctapipe.tools.display_summed_images:main",
    "ctapipe-reconstruct-muons = ctapipe.tools.muon_reconstruction:main",
    "ctapipe-display-integration = ctapipe.tools.display_integrator:main",
    "ctapipe-display-dl1 = ctapipe.tools.display_dl1:main",
    "ctapipe-process = ctapipe.tools.process:main",
    "ctapipe-merge = ctapipe.tools.dl1_merge:main",
    "ctapipe-fileinfo = ctapipe.tools.fileinfo:main",
    "ctapipe-quickstart = ctapipe.tools.quickstart:main",
]
tests_require = ["pytest", "pandas>=0.24.0", "importlib_resources;python_version<'3.9'"]
docs_require = [
    "sphinx_rtd_theme",
    "sphinx_automodapi",
    "sphinx~=3.5",
    "nbsphinx",
    "numpydoc",
    "jupyter",
    "notebook",
    "graphviz",
    "pandas",
]

setup(
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "astropy>=4.0.5,<5",
        "bokeh~=1.0",
        "eventio>=1.5.0,<2.0.0a0",
        "h5py",
        "iminuit>=2",
        "joblib",
        "matplotlib~=3.0",
        "numba>=0.43",
        "numpy~=1.16",
        "psutil",
        "scikit-learn",
        "scipy~=1.2",
        "tables~=3.4",
        "tqdm>=4.32",
        "traitlets~=5.0,>=5.0.5",
        "zstandard",
        "requests",
        "setuptools_scm>=3.4",
        "importlib_resources;python_version<'3.9'",
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
    package_data={
        "": ["tools/bokeh/*.yaml", "tools/bokeh/templates/*.html", "resources/*"]
    },
)
