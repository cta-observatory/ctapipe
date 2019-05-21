# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""

TODO:
- Speed tests, need to be certain the looping on all telescopes is not killing
performance
- Introduce new weighting schemes
- Make intersect_lines code more readable

"""
import numpy as np
import itertools
import astropy.units as u
from ctapipe.reco.reco_algorithms import Reconstructor
from ctapipe.io.containers import ReconstructedShowerContainer
from ctapipe.instrument import get_atmosphere_profile_functions

from astropy.coordinates import SkyCoord, AltAz
from ctapipe.coordinates import NominalFrame
from ctapipe.coordinates import TiltedGroundFrame, project_to_ground

__all__ = [
    'HillasIntersection'
]


class HillasIntersection(Reconstructor):
    """
    This class is a simple re-implementation of Hillas parameter based event
    reconstruction. e.g. https://arxiv.org/abs/astro-ph/0607333

    In this case the Hillas parameters are all constructed in the shared
    angular ( Nominal) system. Direction reconstruction is performed by
    extrapolation of the major axes of the Hillas parameters in the nominal
    system and the weighted average of the crossing points is taken. Core
    reconstruction is performed by performing the same procedure in the
    tilted ground system.

    The height of maximum is reconstructed by the projection os the image
    centroid onto the shower axis, taking the weighted average of all images.

    Uncertainties on the positions are provided by taking the spread of the
    crossing points, however this means that no uncertainty can be provided
    for multiplicity 2 events.
    """

    def __init__(self, atmosphere_profile_name="paranal"):

        # We need a conversion function from height above ground to depth of maximum
        # To do this we need the conversion table from CORSIKA
        _ = get_atmosphere_profile_functions(atmosphere_profile_name)
        self.thickness_profile, self.altitude_profile = _

    def predict(self, hillas_parameters, tel_x, tel_y, array_direction):
        """

        Parameters
        ----------
        hillas_parameters: dict
            Dictionary containing Hillas parameters for all telescopes
            in reconstruction
        tel_x: dict
            Dictionary containing telescope position on ground for all
            telescopes in reconstruction
        tel_y: dict
            Dictionary containing telescope position on ground for all
            telescopes in reconstruction
        array_direction: AltAz
            Pointing direction of the array

        Returns
        -------
        ReconstructedShowerContainer:

        """
        src_x, src_y, err_x, err_y = self.reconstruct_nominal(hillas_parameters)
        core_x, core_y, core_err_x, core_err_y = self.reconstruct_tilted(
            hillas_parameters, tel_x, tel_y)
        err_x *= u.rad
        err_y *= u.rad

        nom = SkyCoord(
            x=src_x * u.rad,
            y=src_y * u.rad,
            frame=NominalFrame(array_direction=array_direction)
        )
        horiz = nom.transform_to(AltAz())

        result = ReconstructedShowerContainer()
        result.alt, result.az = horiz.alt, horiz.az

        tilt = SkyCoord(
            x=core_x * u.m,
            y=core_y * u.m,
            frame=TiltedGroundFrame(pointing_direction=array_direction),
        )
        grd = project_to_ground(tilt)
        result.core_x = grd.x
        result.core_y = grd.y

        x_max = self.reconstruct_xmax(
            nom.x, nom.y,
            tilt.x, tilt.y,
            hillas_parameters,
            tel_x, tel_y,
            90 * u.deg - array_direction.alt,
        )

        result.core_uncert = np.sqrt(core_err_x**2 + core_err_y**2) * u.m

        result.tel_ids = [h for h in hillas_parameters.keys()]
        result.average_intensity = np.mean([h.intensity for h in hillas_parameters.values()])
        result.is_valid = True

        src_error = np.sqrt(err_x**2 + err_y**2)
        result.alt_uncert = src_error.to(u.deg)
        result.az_uncert = src_error.to(u.deg)
        result.h_max = x_max
        result.h_max_uncert = np.nan
        result.goodness_of_fit = np.nan

        return result

    def reconstruct_nominal(self, hillas_parameters, weighting="Konrad"):
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
        if len(hillas_parameters) < 2:
            return None  # Throw away events with < 2 images

        # Find all pairs of Hillas parameters
        combos = itertools.combinations(list(hillas_parameters.values()), 2)
        hillas_pairs = list(combos)

        # Copy parameters we need to a numpy array to speed things up
        h1 = list(
            map(
                lambda h: [h[0].psi.to(u.rad).value,
                           h[0].x.value,
                           h[0].y.value,
                           h[0].intensity], hillas_pairs
            )
        )
        h1 = np.array(h1)
        h1 = np.transpose(h1)

        h2 = list(
            map(lambda h: [h[1].psi.to(u.rad).value,
                           h[1].x.value,
                           h[1].y.value,
                           h[1].intensity], hillas_pairs)
        )
        h2 = np.array(h2)
        h2 = np.transpose(h2)

        # Perform intersection
        sx, sy = self.intersect_lines(h1[1], h1[2], h1[0],
                                      h2[1], h2[2], h2[0])
        if weighting == "Konrad":
            weight_fn = self.weight_konrad
        elif weighting == "HESS":
            weight_fn = self.weight_HESS

        # Weight by chosen method
        weight = weight_fn(h1[3], h2[3])
        # And sin of interception angle
        weight *= self.weight_sin(h1[0], h2[0])

        # Make weighted average of all possible pairs
        x_pos = np.average(sx, weights=weight)
        y_pos = np.average(sy, weights=weight)
        var_x = np.average((sx - x_pos) ** 2, weights=weight)
        var_y = np.average((sy - y_pos) ** 2, weights=weight)

        # Copy into nominal coordinate

        return x_pos, y_pos, np.sqrt(var_x), np.sqrt(var_y)

    def reconstruct_tilted(self, hillas_parameters, tel_x, tel_y,
                           weighting="Konrad"):
        """
        Core position reconstruction by image axis intersection in the tilted
        system

        Parameters
        ----------
        hillas_parameters: dict
            Hillas parameter objects
        tel_x: dict
            Telescope X positions, tilted system
        tel_y: dict
            Telescope Y positions, tilted system
        weighting: str
            Weighting scheme for averaging of crossing points

        Returns
        -------
        (float, float, float, float):
            core position X, core position Y, core uncertainty X,
            core uncertainty X
        """
        if len(hillas_parameters) < 2:
            return None  # Throw away events with < 2 images
        h = list()
        tx = list()
        ty = list()

        # Need to loop here as dict is unordered
        for tel in hillas_parameters.keys():
            h.append(hillas_parameters[tel])
            tx.append(tel_x[tel])
            ty.append(tel_y[tel])

        # Find all pairs of Hillas parameters
        hillas_pairs = list(itertools.combinations(h, 2))
        tel_x = list(itertools.combinations(tx, 2))
        tel_y = list(itertools.combinations(ty, 2))

        tx = np.zeros((len(tel_x), 2))
        ty = np.zeros((len(tel_y), 2))
        for i, _ in enumerate(tel_x):
            tx[i][0], tx[i][1] = tel_x[i][0].value, tel_x[i][1].value
            ty[i][0], ty[i][1] = tel_y[i][0].value, tel_y[i][1].value

        tel_x = np.array(tx)
        tel_y = np.array(ty)

        # Copy parameters we need to a numpy array to speed things up
        h1 = map(lambda h: [h[0].psi.to(u.rad).value, h[0].intensity], hillas_pairs)
        h1 = np.array(list(h1))
        h1 = np.transpose(h1)

        h2 = map(lambda h: [h[1].psi.to(u.rad).value, h[1].intensity], hillas_pairs)
        h2 = np.array(list(h2))
        h2 = np.transpose(h2)

        # Perform intersection
        cx, cy = self.intersect_lines(tel_x[:, 0], tel_y[:, 0], h1[0],
                                      tel_x[:, 1], tel_y[:, 1], h2[0])

        if weighting == "Konrad":
            weight_fn = self.weight_konrad
        elif weighting == "HESS":
            weight_fn = self.weight_HESS

        # Weight by chosen method
        weight = weight_fn(h1[1], h2[1])
        # And sin of interception angle
        weight *= self.weight_sin(h1[0], h2[0])

        # Make weighted average of all possible pairs
        x_pos = np.average(cx, weights=weight)
        y_pos = np.average(cy, weights=weight)
        var_x = np.average((cx - x_pos) ** 2, weights=weight)
        var_y = np.average((cy - y_pos) ** 2, weights=weight)

        return x_pos, y_pos, np.sqrt(var_x), np.sqrt(var_y)

    def reconstruct_xmax(self, source_x, source_y, core_x, core_y,
                         hillas_parameters, tel_x, tel_y, zen):
        """
        Geometrical depth of shower maximum reconstruction, assuming the shower
        maximum lies at the image centroid

        Parameters
        ----------
        source_x: float
            Source X position in nominal system
        source_y: float
            Source Y position in nominal system
        core_x: float
            Core X position in nominal system
        core_y: float
            Core Y position in nominal system
        hillas_parameters: dict
            Dictionary of hillas parameters objects
        tel_x: dict
            Dictionary of telescope X positions
        tel_y: dict
            Dictionary of telescope X positions
        zen: float
            Zenith angle of shower

        Returns
        -------
        float:
            Estimated depth of shower maximum
        """
        cog_x = list()
        cog_y = list()
        amp = list()

        tx = list()
        ty = list()

        # Loops over telescopes in event
        for tel in hillas_parameters.keys():
            cog_x.append(hillas_parameters[tel].x.to(u.rad).value)
            cog_y.append(hillas_parameters[tel].y.to(u.rad).value)
            amp.append(hillas_parameters[tel].intensity)

            tx.append(tel_x[tel].to(u.m).value)
            ty.append(tel_y[tel].to(u.m).value)

        height = get_shower_height(source_x.to(u.rad).value,
                                   source_y.to(u.rad).value,
                                   np.array(cog_x),
                                   np.array(cog_y),
                                   core_x.to(u.m).value,
                                   core_y.to(u.m).value,
                                   np.array(tx),
                                   np.array(ty))
        weight = np.array(amp)
        mean_height = np.sum(height * weight) / np.sum(weight)

        # This value is height above telescope in the tilted system,
        # we should convert to height above ground
        mean_height *= np.cos(zen)

        # Add on the height of the detector above sea level
        mean_height += 2100  # TODO: replace with instrument info

        if mean_height > 100000 or np.isnan(mean_height):
            mean_height = 100000

        mean_height *= u.m
        # Lookup this height in the depth tables, the convert Hmax to Xmax
        x_max = self.thickness_profile(mean_height.to(u.km))
        # Convert to slant depth
        x_max /= np.cos(zen)

        return x_max

    @staticmethod
    def intersect_lines(xp1, yp1, phi1, xp2, yp2, phi2):
        """
        Perform intersection of two lines. This code is borrowed from read_hess.
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
        sin_1 = np.sin(phi1)
        cos_1 = np.cos(phi1)
        a1 = sin_1
        b1 = -1 * cos_1
        c1 = yp1 * cos_1 - xp1 * sin_1

        sin_2 = np.sin(phi2)
        cos_2 = np.cos(phi2)

        a2 = sin_2
        b2 = -1 * cos_2
        c2 = yp2 * cos_2 - xp2 * sin_2

        det_ab = (a1 * b2 - a2 * b1)
        det_bc = (b1 * c2 - b2 * c1)
        det_ca = (c1 * a2 - c2 * a1)

        # if  math.fabs(det_ab) < 1e-14 : # /* parallel */
        #    return 0,0
        xs = det_bc / det_ab
        ys = det_ca / det_ab

        return xs, ys

    @staticmethod
    def weight_konrad(p1, p2):
        return (p1 * p2) / (p1 + p2)

    @staticmethod
    def weight_hess(p1, p2):
        return 1 / ((1 / p1) + (1 / p2))

    @staticmethod
    def weight_sin(phi1, phi2):
        return np.abs(np.sin(np.fabs(phi1 - phi2)))


def get_shower_height(source_x, source_y, cog_x, cog_y,
                      core_x, core_y, tel_pos_x, tel_pos_y):
    """
    Function to calculate the depth of shower maximum geometrically under the assumption
    that the shower maximum lies at the brightest point of the camera image.
    Parameters
    ----------
    source_x: float
        Event source position in nominal frame
    source_y: float
        Event source position in nominal frame
    core_x: float
        Event core position in telescope tilted frame
    core_y: float
        Event core position in telescope tilted frame
    zen: float
        Zenith angle of event
    Returns
    -------
    float: Depth of maximum of air shower
    """

    # Calculate displacement of image centroid from source position (in rad)
    disp = np.sqrt(np.power(cog_x - source_x, 2) +
                   np.power(cog_y - source_y, 2))
    # Calculate impact parameter of the shower
    impact = np.sqrt(np.power(tel_pos_x - core_x, 2) +
                     np.power(tel_pos_y - core_y, 2))

    # Distance above telescope is ration of these two (small angle)
    height = impact / disp

    return height
