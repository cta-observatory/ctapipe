# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Coordinates.
"""


from .coordinate_transformations import (pixel_position_to_direction,
                                         alt_to_theta, az_to_phi,
                                         transform_pixel_position)


from .angular_frames import *
from .ground_frames import *


from astropy.utils import iers
iers.conf.auto_download = True  # auto-fetch updates to IERS_A table
