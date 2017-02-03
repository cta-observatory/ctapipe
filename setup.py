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

PACKAGENAME = metadata.get('package_name', 'packagename')
DESCRIPTION = metadata.get('description', 'Astropy affiliated package')
AUTHOR = metadata.get('author', '')
AUTHOR_EMAIL = metadata.get('author_email', '')
LICENSE = metadata.get('license', 'unknown')
URL = metadata.get('url', 'http://astropy.org')

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
    'ctapipe-flow = ctapipe.flow.flow:main'
]

package.version.update_release_version()

setup(name=PACKAGENAME,
      packages=find_packages(),
      version=package.version.get_version(pep440=True),
      description=DESCRIPTION,
      # these should be minimum list of what is needed to run (note
      # don't need to list the sub-dependencies like numpy, since
      # astropy already depends on it)
      install_requires=['astropy', 'scipy', 'matplotlib',
                        'scikit-learn', 'traitlets'],
#      setup_requires=[, ],
      tests_require=['pytest', ],
      extras_require={
        'dev': [
            'pytest',
            'pytest-pep8',
            'pytest-cov',
            'sphinx',
            'sphinx_rtd_theme',
            'sphinx-automodapi',
            'graphviz',
            'numpydoc',
            
        ]
      },
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url=URL,
      long_description=LONG_DESCRIPTION,
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: C',
          'Programming Language :: Cython',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: Implementation :: CPython',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Development Status :: 3 - Alpha',
      ],
      zip_safe=False,
      use_2to3=False,
      entry_points=entry_points,
      )
