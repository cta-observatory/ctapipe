# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""



"""
import numpy as np
import itertools
import astropy.units as u
from ctapipe.reco.reco_algorithms import RecoShowerGeomAlgorithm
from ctapipe.io.containers import ReconstructedShowerContainer
from ctapipe.coordinates import *

__all__ = [
    'HillasIntersection'
]


class HillasIntersection(RecoShowerGeomAlgorithm):
    '''


    '''

    def predict(self, hillas_parameters, tel_x, tel_y, array_direction):
        """

        Parameters
        ----------
        hillas_parameters
        tel_x
        tel_y

        Returns
        -------

        """
        src_x, src_y, err_x, err_y = self.reconstruct_nominal(hillas_parameters)
        core_x, core_y, core_err_x, core_err_y = self.reconstruct_tilted(hillas_parameters, tel_x, tel_y)
        err_x *= u.rad
        err_y *= u.rad

        nom = NominalFrame(x=src_x*u.rad, y=src_y*u.rad, array_direction=array_direction)
        horiz = nom.transform_to(HorizonFrame())
        result = ReconstructedShowerContainer()
        result.alt, result.az = horiz.alt, horiz.az
        result.core_x = core_x * u.m
        result.core_y = core_y * u.m

        result.core_uncert = np.sqrt(core_err_x*core_err_x + core_err_y*core_err_y) * u.m

        result.tel_ids = [h for h in hillas_parameters.keys()]
        result.average_size = np.mean([h.size for h in hillas_parameters.values()])
        result.is_valid = True

        src_error = np.sqrt(err_x*err_x + err_y*err_y)
        result.alt_uncert = src_error.to(u.deg)
        result.az_uncert = src_error.to(u.deg)
        result.h_max = np.nan
        result.h_max_uncert = np.nan
        result.goodness_of_fit = np.nan

        return result

    def reconstruct_nominal(self, hillas_parameters,weighting="Konrad"):
        """
        Perform event reconstruction by simple Hillas parameter intersection
        in the nominal system

        Parameters
        ----------
        hillas_parameters: dict
            Hillas parameter objects
        weighting: string
            Specify image weighting scheme used (HESS or Konrad style)

        Returns
        -------
        Reconstructed event position in the nominal system

        """
        if len(hillas_parameters)<2:
            return None # Throw away events with < 2 images

        # Find all pairs of Hillas parameters
        hillas_pairs = list(itertools.combinations(list(hillas_parameters.values()), 2))

        # Copy parameters we need to a numpy array to speed things up
        h1 = list(map(lambda h:[h[0].psi.to(u.rad).value,h[0].cen_x.value,h[0].cen_y.value,h[0].size],hillas_pairs))
        h1 = np.array(h1)
        h1 = np.transpose(h1)

        h2 = np.array(list(map(lambda h:[h[1].psi.to(u.rad).value,h[1].cen_x.value,h[1].cen_y.value,h[1].size],hillas_pairs)))
        h2 = np.array(h2)
        h2 = np.transpose(h2)

        # Perform intersection
        sx,sy = self.intersect_lines(h1[1],h1[2],h1[0],
                                     h2[1],h2[2],h2[0])
        if weighting == "Konrad":
            weight_fn = self.weight_konrad
        elif weighting == "HESS":
            weight_fn = self.weight_HESS

        weight = weight_fn(h1[3],h2[3])
        weight *= self.weight_sin(h1[0],h2[0])

        # Make weighted average of all possible pairs
        x_pos = np.average(sx, weights=weight)
        y_pos = np.average(sy, weights=weight)
        var_x = np.average((sx - x_pos) ** 2, weights=weight)
        var_y = np.average((sy - y_pos) ** 2, weights=weight)

        # Copy into nominal coordinate

        return x_pos, y_pos, np.sqrt(var_x), np.sqrt(var_y)

    def reconstruct_tilted(self, hillas_parameters,tel_x,tel_y,weighting="Konrad"):
        """

        Parameters
        ----------
        hillas_parameters
        tel_x
        tel_y
        weighting

        Returns
        -------

        """
        if len(hillas_parameters)<2:
            return None # Throw away events with < 2 images

        # Find all pairs of Hillas parameters
        hillas_pairs = list(itertools.combinations(list(hillas_parameters.values()), 2))
        tel_x = list(itertools.combinations(list(tel_x.values()), 2))
        tel_y= list(itertools.combinations(list(tel_y.values()), 2))

        tx = np.zeros((len(tel_x),2))
        ty = np.zeros((len(tel_y),2))
        for i in range(len(tel_x)):
            tx[i][0], tx[i][1] = tel_x[i][0].value, tel_x[i][1].value
            ty[i][0], ty[i][0] = tel_y[i][0].value, tel_y[i][1].value

        tel_x = np.array(tx)
        tel_y = np.array(ty)

         # Copy parameters we need to a numpy array to speed things up
        h1 = list(map(lambda h:[h[0].psi.to(u.rad).value,h[0].size],hillas_pairs))
        h1 = np.array(h1)
        h1 = np.transpose(h1)

        h2 = np.array(list(map(lambda h:[h[1].psi.to(u.rad).value,h[1].size],hillas_pairs)))
        h2 = np.array(h2)
        h2 = np.transpose(h2)
        # Perform intersection
        cx,cy = self.intersect_lines(tel_x[:,0],tel_y[:,0],h1[0],
                                     tel_x[:,1],tel_y[:,1],h2[0])

        if weighting == "Konrad":
            weight_fn = self.weight_konrad
        elif weighting == "HESS":
            weight_fn = self.weight_HESS

        weight = weight_fn(h1[1],h2[1])
        weight *= self.weight_sin(h1[0],h2[0])

        # Make weighted average of all possible pairs
        x_pos = np.average(cx, weights=weight)
        y_pos = np.average(cy, weights=weight)
        var_x = np.average((cx - x_pos) ** 2, weights=weight)
        var_y = np.average((cy - y_pos) ** 2, weights=weight)

        return x_pos, y_pos, np.sqrt(var_x), np.sqrt(var_y)

    @staticmethod
    def intersect_lines(xp1,yp1,phi1,xp2,yp2,phi2):
        """

        Parameters
        ----------
        xp1: ndarray
            X position of first image
        yp1: ndarray
            Y position of first image
        phi1: ndarray
            Rotation angle of first image
        xp2: ndarray
            X position of second image
        yp2: ndarray
            Y position of second image
        phi2: ndarray
            Rotation angle of second image

        Returns
        -------
        ndarray of x and y crossing points for all pairs
        """
        s1 = np.sin(phi1)
        c1 = np.cos(phi1)
        A1 = s1
        B1 = -1*c1
        C1 = yp1*c1 - xp1*s1

        s2 = np.sin(phi2)
        c2 = np.cos(phi2)

        A2 = s2
        B2 = -1*c2
        C2 = yp2*c2 - xp2*s2

        detAB = (A1*B2-A2*B1)
        detBC = (B1*C2-B2*C1)
        detCA = (C1*A2-C2*A1)

        #if  math.fabs(detAB) < 1e-14 : # /* parallel */
        #    return 0,0
        xs = detBC / detAB
        ys = detCA / detAB

        return xs,ys

    @staticmethod
    def weight_konrad(p1,p2):
        return (p1*p2)/(p1+p2)

    @staticmethod
    def weight_HESS(p1,p2):
        return 1/((1/p1)+(1/p2))

    @staticmethod
    def weight_sin(phi1,phi2):
        return np.abs(np.sin(np.fabs(phi1-phi2)))