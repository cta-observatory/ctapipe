# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Coordinates.
"""


from astropy.utils import iers
iers.conf.auto_download = True # auto-fetch updates to IERS_A table


from .angular_frames import *
from .ground_frames import *
