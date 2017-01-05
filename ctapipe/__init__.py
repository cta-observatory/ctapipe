# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
ctapipe - CTA Python pipeline experimental version
"""

from . import version as v
__version__ = v.get_version(pep440=False)
version = v.get_version(pep440=False)
