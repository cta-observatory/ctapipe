# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Camera calibration module.
"""

from .calibrator import CameraCalibrator  # noqa: F401
from .gainselection import GainSelector  # noqa: F401

# CameraCalibrator and GainSelector not added to __all__ here to avoid
# docs build error as it would result in the class being exposed at 3 levels
# which is not supported by sphinx
__all__ = []
