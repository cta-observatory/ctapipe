# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Hillas shower parametrization.



"""
import numpy as np
import astropy.units as u
from ctapipe.coordinates.frames import NominalFrame,TiltedGroundFrame,GroundFrame

__all__ = [
    'reconstruct_event'
]


def reconstruct_event(hillas_parameters,telescope_positions,pixel_list,shower_seed=None):
    """
    Perform event reconstruction

    Parameters
    ----------
    hillas_parameters: list
        Hillas parameter objects
    telescope_positions: list
        XY positions of telescopes (ground system)
    shower_seed: shower object?
        Seed position to begin shower minimisation

    Returns
    -------
    Reconstructed event position in the nominal system

    """
    if len(hillas_parameters)<2:
        return None # Throw away events with < 2 images

    return None


def rotate_translate(pixel_pos_x,pixel_pos_y,x_trans,y_trans,phi):
    """
    Function to perform rotation and translation of pixel lists

    Parameters
    ----------
    pixel_pos_x: ndarray
        Array of pixel x positions
    pixel_pos_y: ndarray
        Array of pixel x positions
    x_trans: float
        Translation of position in x coordinates
    y: float
        Translation of position in y coordinates
    phi: float
        Rotation angle of pixels

    Returns
    -------
        ndarray,ndarray: Transformed pixel x and y coordinates
    """
    pixel_pos_trans_x = (pixel_pos_x-x_trans) *np.cos(phi) - (pixel_pos_y-y_trans)*np.sin(phi)
    pixel_pos_trans_y = (pixel_pos_x-x_trans) *np.sin(phi) + (pixel_pos_y-y_trans)*np.cos(phi)

    return pixel_pos_trans_x,pixel_pos_trans_y