# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Camera calibration module.
"""

from .calibrator import CameraCalibrator  # noqa: F401
from .gainselection import GainSelector  # noqa: F401

# CameraCalibrator, GainSelector are not in __all__ here to prevent documentation
# at this level, as it would break sphinx by creating three exports of the same class
__all__ = []
