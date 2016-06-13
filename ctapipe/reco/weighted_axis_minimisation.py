# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Hillas shower parametrization.



"""
import numpy as np
import astropy.units as u
from iminuit import Minuit

__all__ = [
    'reconstruct_event'
]

class weighted_axis_minimisation:

    def reconstruct_event(self,hillas_parameters,telescope_pos_x,telescope_pos_y,pixel_pos_x,pixel_pos_y,pixel_weight=1,shower_seed=None):
        """
        Perform event reconstruction

        Parameters
        ----------
        hillas_parameters: list
            Hillas parameter objects
        telescope_positions: list
            XY positions of telescopes (tilted system)
        shower_seed: shower object?
            Seed position to begin shower minimisation

        Returns
        -------
        Reconstructed event position in the nominal system

        """
        if len(hillas_parameters)<2:
            return None # Throw away events with < 2 images

        self.pixel_pos_x = pixel_pos_x
        self.pixel_pos_y = pixel_pos_y
        self.pixel_weight = pixel_weight
        self.tel_pos_x = telescope_pos_x
        self.tel_pos_y = telescope_pos_y

        if shower_seed is None:
            x_src=0
            y_src=0
            x_grd=0
            y_grd=0

        m = Minuit(self.weighted_dist,x=0,error_x=1,y=1,error_y=1,z=1,error_z=1,errordef=1)
        m.migrad()

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
        y_trans: float
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

    def get_dist_from_axis(self,pixel_pos_x,pixel_pos_y,x_trans,y_trans,phi,pixel_weight):
        """
        Function to perform weighting of pixel positions from the shower axis for an individual image

        Parameters
        ----------
        pixel_pos_x: ndarray
            Array of pixel x positions
        pixel_pos_y: ndarray
            Array of pixel x positions
        x_trans: float
            Translation of position in x coordinates
        y_trans: float
            Translation of position in y coordinates
        phi: float
            Rotation angle of pixels
        pixel_weight: ndarray
            Weighting of individual pixels (usually amplitude)

        Returns
        -------
            Weighted sum of distances of pixels from shower axis
        """
        pixel_pos_x_trans,pixel_pos_y_trans = self.rotate_translate(pixel_pos_x,pixel_pos_y,x_trans,y_trans,phi)
        pixel_pos_y_trans *= pixel_weight

        return np.sum(pixel_pos_y_trans)

    def weighted_dist(self,x_src,y_src,x_grd,y_grd):

        sum = 0
        for pos_x,pos_y,tel_x,tel_y,weight in self.pixel_pos_x,self.pixel_pos_y,self.tel_pos_x,self.tel_pos_y,self.pixel_weight:
            phi = np.atan((tel_y-y_grd)/(tel_x-x_grd))
            sum += self.get_dist_from_axis(pos_x,pos_y,x_src,y_src,phi,weight)
        return sum



