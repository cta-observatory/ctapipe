#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import ah_bootstrap
import pkg_resources
from setuptools import setup
from astropy_helpers.setup_helpers import register_commands, get_package_info
from astropy_helpers.version_helpers import generate_version_py

NAME = 'astropy_helpers'
VERSION = '1.0.2'
RELEASE = 'dev' not in VERSION
DOWNLOAD_BASE_URL = 'http://pypi.python.org/packages/source/a/astropy-helpers'

generate_version_py(NAME, VERSION, RELEASE, False, uses_git=not RELEASE)

# Use the updated version including the git rev count
from astropy_helpers.version import version as VERSION

cmdclass = register_commands(NAME, VERSION, RELEASE)
# This package actually doesn't use the Astropy test command
del cmdclass['test']

setup(
    name=pkg_resources.safe_name(NAME),  # astropy_helpers -> astropy-helpers
    version=VERSION,
    description='Utilities for building and installing Astropy, Astropy '
                'affiliated packages, and their respective documentation.',
    author='The Astropy Developers',
    author_email='astropy.team@gmail.com',
    license='BSD',
    url='http://astropy.org',
    long_description=open('README.rst').read(),
    download_url='{0}/astropy-helpers-{1}.tar.gz'.format(DOWNLOAD_BASE_URL,
                                                         VERSION),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Framework :: Setuptools Plugin',
        'Framework :: Sphinx :: Extension',
        'Framework :: Sphinx :: Theme',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Archiving :: Packaging'
    ],
    cmdclass=cmdclass,
    zip_safe=False,
    **get_package_info(exclude=['astropy_helpers.tests'])
)
