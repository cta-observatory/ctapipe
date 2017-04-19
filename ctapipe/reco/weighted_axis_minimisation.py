# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Event reconstruction is nominal and ground system.

The method uses a simple algorithm described in:
http://arxiv.org/pdf/astro-ph/9904234v1.pdf  (algorithm 6)

This method determines the best fit source and core position by minimising
the distance between the predicted shower plane and the pixels in the image
(weighted by pixel amplitude). This is not a very powerful method, but serves
as an example for more complex and powerful minimisation based methods.

In this case this must be implemented as a class to work properly with the
iMinuit implementation

ToDo:
    - Standardise inputs and outputs

"""
import numpy as np
import astropy.units as u
from iminuit import Minuit
import matplotlib.pyplot as plt
import matplotlib.cm as cm

__all__ = [
    'WeightedAxisMinimisation'
]


class WeightedAxisMinimisation:
    def __init__(self):
        return None

    def reconstruct_event(self,hillas_parameters,telescope_pos_x,telescope_pos_y,pixel_pos_x,pixel_pos_y,pixel_weight=1,
                          shower_seed=None,draw=False):
        """
        Perform event reconstruction

        Parameters
        ----------
        hillas_parameters: list
            Hillas parameter objects
        telescope_pos_x: list
            X position of telescopes (tilted system)
        telescope_pos_y: list
            Y position of telescopes (tilted system)
        pixel_pos_x: list
            X position of image pixels (nominal system)
        pixel_pos_y: list
            Y position of image pixels (nominal system)
        pixel_weight: list
            Weighting of each pixel in fit (usually amplitude)
        shower_seed: shower object?
            Seed position to begin shower minimisation
        draw: boolean
            Control drawing of likelihood surface

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


        x_src=0.
        y_src=0.
        x_grd=shower_seed[0]
        y_grd=shower_seed[1]

        m = Minuit(self.weighted_dist,x_src=x_src,error_x_src=0.1,y_src=y_src,error_y_src=0.1,
                   x_grd=x_grd,error_x_grd=10,y_grd=y_grd,error_y_grd=10,errordef=1)
        m.migrad()

        if draw:
            self.draw_surfaces(m.values['x_src'],m.values['y_src'],
                                m.values['x_grd'],m.values['y_grd'])

        plt.show()

        return m.values

    @staticmethod
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
        return np.sum(np.abs(pixel_pos_y_trans))

    def weighted_dist(self,x_src,y_src,x_grd,y_grd):
        """
        Function to be minimised to find core and source position.
        Calculates expected image axis in nominal system and returns the weighted sum
        of distances from the predicted image axis.

        Parameters
        ----------
        x_src: float
            Test source position in nominal system
        y_src: float
            Test source position in nominal system
        x_grd: float
            Test core position in tilted system
        y_grd: float
            Test core position in tilted system

        Returns
        -------
            Sum of weighted pixel distances from predicted axis
        """

        sum = 0
        for pos_x,pos_y,tel_x,tel_y,weight in zip(self.pixel_pos_x,self.pixel_pos_y,self.tel_pos_x,self.tel_pos_y,self.pixel_weight):
            phi = np.arctan2((tel_y-y_grd),(tel_x-x_grd)) * u.rad
            sum += self.get_dist_from_axis(pos_x,pos_y,x_src,y_src,phi,weight)

        return sum

    def draw_surfaces(self,x_src,y_src,x_grd,y_grd):
        """
        Simple function to draw the surface of the test statistic in both the nominal
        and tilted planes while keeping the values in the other plane fixed at the best fit
        value.

        Parameters
        ----------
        x_src: float
            Source position in nominal coordinates (centre of map)
        y_src: float
            Source position in nominal coordinates (centre of map)
        x_grd: float
            Ground position in tilted coordinates (centre of map)
        y_grd: float
            Ground position in tilted coordinates (centre of map)

        Returns
        -------

        """
        fig = plt.figure(figsize=(12, 12))
        nom1 = fig.add_subplot(221)
        self.draw_nominal_surface(x_src,y_src,x_grd,y_grd,nom1,bins=25,range=0.2)
        nom1.plot(x_src,y_src,"wo")

        nom2 = fig.add_subplot(222)
        self.draw_nominal_surface(x_src,y_src,x_grd,y_grd,nom2,bins=25,range=5)
        nom2.plot(x_src,y_src,"wo")

        tilt1 = fig.add_subplot(223)
        self.draw_tilted_surface(x_src,y_src,x_grd,y_grd,tilt1,bins=25,range=100)
        tilt1.plot(self.tel_pos_x,self.tel_pos_y,"ro")
        tilt1.plot(x_grd,y_grd,"wo")

        tilt2 = fig.add_subplot(224)
        self.draw_tilted_surface(x_src,y_src,x_grd,y_grd,tilt2,bins=25,range=500)
        tilt2.plot(self.tel_pos_x,self.tel_pos_y,"ro")
        tilt2.plot(x_grd,y_grd,"wo")

        plt.show()

        return

    def draw_nominal_surface(self,x_src,y_src,x_grd,y_grd,plot_name,bins=100,range=1):
        """
        Function for creating test statistic surface in nominal plane

        Parameters
        ----------
        x_src: float
            Source position in nominal coordinates (centre of map)
        y_src: float
            Source position in nominal coordinates (centre of map)
        x_grd: float
            Ground position in tilted coordinates (centre of map)
        y_grd: float
            Ground position in tilted coordinates (centre of map)
        plot_name: matplotlib axis
            Subplot in which to include this plot
        bins: int
            Number of bins in each axis
        range: float
            Size of map

        Returns
        -------
            None
        """
        x_dir = np.linspace(x_src-range,x_src+range,num=bins)
        y_dir = np.linspace(y_src-range,y_src+range,num=bins)
        xd = list()
        yd = list()
        w = list()

        for xb in x_dir:
            for yb in y_dir:
                xd.append(xb)
                yd.append(yb)
                w.append(self.weighted_dist(xb,yb,x_grd,y_grd))
        h, x, y, p = plt.hist2d(xd, yd, bins=bins, weights=w)
        return plot_name.imshow(h, interpolation = "bilinear", cmap=plt.cm.viridis_r,vmin=np.min(h),vmax=np.max(h),
                                extent=(x_src-range,x_src+range,y_src-range,y_src+range))

    def draw_tilted_surface(self,x_src,y_src,x_grd,y_grd,plot_name,bins=100,range=100):
        """
        Function for creating test statistic surface in tilted plane

        Parameters
        ----------
        x_src: float
            Source position in nominal coordinates (centre of map)
        y_src: float
            Source position in nominal coordinates (centre of map)
        x_grd: float
            Ground position in tilted coordinates (centre of map)
        y_grd: float
            Ground position in tilted coordinates (centre of map)
        plot_name: matplotlib axis
            Subplot in which to include this plot
        bins: int
            Number of bins in each axis
        range: float
            Size of map

        Returns
        -------
            None
        """
        x_ground_list = np.linspace(x_grd-range,x_grd+range,num=bins)
        y_ground_list = np.linspace(y_grd-range,y_grd+range,num=bins)
        xd = list()
        yd = list()
        w = list()

        for xb in x_ground_list:
            for yb in y_ground_list:
                xd.append(xb)
                yd.append(yb)
                w.append(self.weighted_dist(x_src,y_src,xb,yb))

        h, x, y, p = plt.hist2d(xd, yd, bins=bins, weights=w)
        return plot_name.imshow(h, interpolation = "bilinear", cmap=plt.cm.viridis_r,vmin=np.min(h),vmax=np.max(h),
                                extent=(x_grd-range,x_grd+range,y_grd-range,y_grd+range))