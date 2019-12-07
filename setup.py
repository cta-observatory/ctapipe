#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import sys

# import ah_bootstrap
from setuptools import setup, find_packages

# Get some values from the setup.cfg
from configparser import RawConfigParser
conf = RawConfigParser()
conf.read(['setup.cfg'])
metadata = dict(conf.items('metadata'))

PACKAGENAME = metadata['package_name']
DESCRIPTION = metadata['description']
AUTHOR = metadata['author']
AUTHOR_EMAIL = metadata['author_email']
LICENSE = metadata['license']
URL = metadata['url']

# Get the long description from the package's docstring
__import__(PACKAGENAME)
package = sys.modules[PACKAGENAME]
LONG_DESCRIPTION = package.__doc__

# Define entry points for command-line scripts
# TODO: this shuold be automated (e.g. look for main functions and
# rename _ to -, and prepend 'ctapipe'
entry_points = {}
entry_points['console_scripts'] = [
    'ctapipe-info = ctapipe.tools.info:main',
    'ctapipe-camdemo = ctapipe.tools.camdemo:main',
    'ctapipe-dump-triggers = ctapipe.tools.dump_triggers:main',
    'ctapipe-chargeres-extract = ctapipe.tools.extract_charge_resolution:main',
    'ctapipe-chargeres-plot = ctapipe.tools.plot_charge_resolution:main',
    'ctapipe-dump-instrument=ctapipe.tools.dump_instrument:main',
    'ctapipe-event-viewer = ctapipe.tools.bokeh.file_viewer:main',
    'ctapipe-display-tel-events = ctapipe.tools.display_events_single_tel:main',
    'ctapipe-display-imagesums = ctapipe.tools.display_summed_images:main',
    'ctapipe-reconstruct-muons = ctapipe.tools.muon_reconstruction:main',
    'ctapipe-display-integration = ctapipe.tools.display_integrator:main',
    'ctapipe-display-dl1 = ctapipe.tools.display_dl1:main',
]
tests_require = [
    'pytest',
    'ctapipe-extra @ https://github.com/cta-observatory/ctapipe-extra/archive/v0.2.18.tar.gz',
    'pyhessio @ https://github.com/cta-observatory/pyhessio/archive/v2.1.1.tar.gz',
]
docs_require = [
    'sphinx_rtd_theme', 'sphinx_automodapi', 'sphinx', 'nbsphinx', 'numpydoc',
    'jupyter', 'notebook', 'travis-sphinx', 'graphviz',
]

package.version.update_release_version()

setup(name=PACKAGENAME,
      packages=find_packages(),
      version=package.version.get_version(pep440=True),
      description=DESCRIPTION,
      python_requires='>=3.6',
      # these should be minimum list of what is needed to run (note
      # don't need to list the sub-dependencies like numpy, since
      # astropy already depends on it)
      install_requires=[
          'astropy~=3.0',
          'bokeh~=1.0',
          'eventio~=1.0',
          'iminuit>=1.3',
          'joblib',
          'matplotlib~=3.0',
          'numba>=0.43',
          'numpy~=1.11',
          'pandas>=0.24.0',
          'psutil',
          'scikit-learn',
          'scipy~=1.2',
          'tables~=3.4',
          'tqdm>=4.32',
          'traitlets>=4.1,<5.0',
      ],
      # here are optional dependencies (as "tag" : "dependency spec")
      extras_require={
          'all': tests_require + docs_require,
          'tests': tests_require,
          'docs': docs_require,
      },
      tests_require=tests_require,
      setup_requires=['pytest_runner'],
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url=URL,
      long_description=LONG_DESCRIPTION,
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: Implementation :: CPython',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Development Status :: 3 - Alpha',
      ],
      zip_safe=False,
      use_2to3=False,
      entry_points=entry_points,
      package_data={
          '': ['tools/bokeh/*.yaml', 'tools/bokeh/templates/*.html'],
      }
      )
