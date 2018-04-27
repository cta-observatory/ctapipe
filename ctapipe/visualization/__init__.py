# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Visualization: Methods for displaying data 
"""

try:
    from .mpl_camera import CameraDisplay
    from .mpl_array import ArrayDisplay
except ImportError:
    pass


__all__ = ['CameraDisplay', 'ArrayDisplay']
