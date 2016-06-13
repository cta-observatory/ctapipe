# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Hillas shower parametrization.



"""
import numpy as np
import astropy.units as u
from ctapipe.coordinates.frames import NominalFrame,TiltedGroundFrame,GroundFrame

__all__ = [
    'reconstruct_event'
]


def reconstruct_event(hillas_parameters,telescope_positions,shower_seed=None):
    """
    Perform event reconstruction

    Parameters
    ----------
    hillas_parameters: list
        Hillas parameter objects
    telescope_positions: list
        XY positions of telescopes (ground system)

    Returns
    -------
    Reconstructed event position in the nominal system

    """
    if len(hillas_parameters)<2:
        return None # Throw away events with < 2 images

    return None
