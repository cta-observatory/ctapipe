# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""

TODO:
- Speed tests, need to be certain the looping on all telescopes is not killing
performance
- Introduce new weighting schemes
- Make intersect_lines code more readable

"""
import itertools
import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, SkyCoord

from ..containers import (
    CameraHillasParametersContainer,
    HillasParametersContainer,
    ReconstructedGeometryContainer,
)
from ..coordinates import (
    CameraFrame,
    MissingFrameAttributeWarning,
    NominalFrame,
    TelescopeFrame,
    TiltedGroundFrame,
    project_to_ground,
)
from ..core import traits
from .reconstructor import (
    HillasGeometryReconstructor,
    InvalidWidthException,
    ReconstructionProperty,
    TooFewTelescopesException,
)

__all__ = ["HillasIntersection"]


INVALID = ReconstructedGeometryContainer(
    telescopes=[],
    prefix="HillasIntersection",
)

FOV_ANGULAR_DISTANCE_LIMIT_RAD = (45 * u.deg).to_value(u.rad)
H_MAX_UPPER_LIMIT_M = 100_000


def _far_outside_fov(fov_lat, fov_lon):
    """
    Check if a given latitude or longiude in the FoV is further away from
    the FoV center than `FOV_ANGULAR_DISTANCE_LIMIT`

    Parameters
    ----------
    fov_lat : u.Quantity[angle]
        Latitude in TelescopeFrame or NominalFrame
    fov_lon : u.Quantity[angle]
        Longitude in TelescopeFrame or NominalFrame

    Returns
    -------
    bool
    """
    lat_outside_fov = np.abs(fov_lat) > FOV_ANGULAR_DISTANCE_LIMIT_RAD
    lon_outside_fov = np.abs(fov_lon) > FOV_ANGULAR_DISTANCE_LIMIT_RAD
    return lat_outside_fov or lon_outside_fov


class HillasIntersection(HillasGeometryReconstructor):
    """
    This class is a simple re-implementation of Hillas parameter based event
    reconstruction. See algorithm I of :cite:p:`hofmann-1999-comparison`.

    In this case the Hillas parameters are all constructed in the shared
    angular (Nominal) system. Direction reconstruction is performed by
    extrapolation of the major axes of the Hillas parameters in the nominal
    system and the weighted average of the crossing points is taken. Core
    reconstruction is performed by performing the same procedure in the
    tilted ground system.

    The height of maximum is reconstructed by the projection of the image
    centroid onto the shower axis, taking the weighted average of all images.

    Uncertainties on the positions are provided by taking the spread of the
    crossing points, however this means that no uncertainty can be provided
    for multiplicity 2 events.
    """

    weighting = traits.CaselessStrEnum(
        ["Konrad", "hess"], default_value="Konrad", help="Weighting Method name"
    ).tag(config=True)

    property = ReconstructionProperty.GEOMETRY

    def __init__(self, subarray, atmosphere_profile=None, **kwargs):
        """
        Weighting must be a function similar to the weight_konrad already implemented
        """
        super().__init__(subarray, atmosphere_profile, **kwargs)

        # We need a conversion function from height above ground to depth of maximum
        # To do this we need the conversion table from CORSIKA

        # other weighting schemes can be implemented. just add them as additional methods
        if self.weighting == "Konrad":
            self._weight_method = self.weight_konrad

    def __call__(self, event):
        """
        Perform stereo reconstruction on event.

        Parameters
        ----------
        event : `~ctapipe.containers.ArrayEventContainer`
            The event, needs to have dl1 parameters.
            Will be filled with the corresponding dl2 containers,
            reconstructed stereo geometry and telescope-wise impact position.
        """

        try:
            hillas_dict = self._create_hillas_dict(event)
        except (TooFewTelescopesException, InvalidWidthException):
            event.dl2.stereo.geometry[self.__class__.__name__] = INVALID
            self._store_impact_parameter(event)
            return

        # Due to tracking the pointing of the array will never be a constant
        array_pointing = SkyCoord(
            az=event.pointing.array_azimuth,
            alt=event.pointing.array_altitude,
            frame=AltAz(),
        )

        telescope_pointings = self._get_telescope_pointings(event)

        event.dl2.stereo.geometry[self.__class__.__name__] = self._predict(
            hillas_dict, array_pointing, telescope_pointings
        )

        self._store_impact_parameter(event)

    def _predict(self, hillas_dict, array_pointing, telescopes_pointings=None):
        """

        Parameters
        ----------
        hillas_dict: dict
            Dictionary containing Hillas parameters for all telescopes
            in reconstruction
        inst : ctapipe.io.InstrumentContainer
            instrumental description
        array_pointing: SkyCoord[AltAz]
            pointing direction of the array
        telescopes_pointings: dict[SkyCoord[AltAz]]
            dictionary of pointing direction per each telescope

        Returns
        -------
        ReconstructedGeometryContainer:

        """

        # filter warnings for missing obs time. this is needed because MC data has no obs time
        warnings.filterwarnings(action="ignore", category=MissingFrameAttributeWarning)

        # stereoscopy needs at least two telescopes
        if len(hillas_dict) < 2:
            raise TooFewTelescopesException(
                "need at least two telescopes, have {}".format(len(hillas_dict))
            )

        # check for np.nan or 0 width's as these screw up weights
        if any([np.isnan(h.width.value) for h in hillas_dict.values()]):
            raise InvalidWidthException(
                "A HillasContainer contains an ellipse of width==np.nan"
            )

        if any([h.width.value == 0 for h in hillas_dict.values()]):
            raise InvalidWidthException(
                "A HillasContainer contains an ellipse of width==0"
            )

        if telescopes_pointings is None:
            telescopes_pointings = {
                tel_id: array_pointing for tel_id in hillas_dict.keys()
            }

        tilted_frame = TiltedGroundFrame(pointing_direction=array_pointing)
        grd_coord = self.subarray.tel_coords
        tilt_coord = grd_coord.transform_to(tilted_frame)

        tel_ids = list(hillas_dict.keys())
        tel_indices = self.subarray.tel_ids_to_indices(tel_ids)

        tel_x = {
            tel_id: tilt_coord.x[tel_index]
            for tel_id, tel_index in zip(tel_ids, tel_indices)
        }
        tel_y = {
            tel_id: tilt_coord.y[tel_index]
            for tel_id, tel_index in zip(tel_ids, tel_indices)
        }

        nom_frame = NominalFrame(origin=array_pointing)

        hillas_dict_mod = {}

        for tel_id, hillas in hillas_dict.items():
            if isinstance(hillas, CameraHillasParametersContainer):
                focal_length = self.subarray.tel[tel_id].optics.equivalent_focal_length
                camera_frame = CameraFrame(
                    telescope_pointing=telescopes_pointings[tel_id],
                    focal_length=focal_length,
                )
                cog_coords = SkyCoord(x=hillas.x, y=hillas.y, frame=camera_frame)
                cog_coords_nom = cog_coords.transform_to(nom_frame)
            else:
                telescope_frame = TelescopeFrame(
                    telescope_pointing=telescopes_pointings[tel_id]
                )
                cog_coords = SkyCoord(
                    fov_lon=hillas.fov_lon,
                    fov_lat=hillas.fov_lat,
                    frame=telescope_frame,
                )
                cog_coords_nom = cog_coords.transform_to(nom_frame)
            hillas_dict_mod[tel_id] = HillasParametersContainer(
                fov_lon=cog_coords_nom.fov_lon,
                fov_lat=cog_coords_nom.fov_lat,
                psi=hillas.psi,
                width=hillas.width,
                length=hillas.length,
                intensity=hillas.intensity,
            )

        src_fov_lon, src_fov_lat, err_fov_lon, err_fov_lat = self.reconstruct_nominal(
            hillas_dict_mod
        )

        # Catch events reconstructed at great angular distance from camera center
        # and return INVALID container to prevent SkyCoord error below.
        if _far_outside_fov(src_fov_lat, src_fov_lon):
            return INVALID

        core_x, core_y, core_err_x, core_err_y = self.reconstruct_tilted(
            hillas_dict_mod, tel_x, tel_y
        )

        err_fov_lon *= u.rad
        err_fov_lat *= u.rad

        nom = SkyCoord(
            fov_lon=src_fov_lon * u.rad, fov_lat=src_fov_lat * u.rad, frame=nom_frame
        )
        sky_pos = nom.transform_to(array_pointing.frame)
        tilt = SkyCoord(x=core_x * u.m, y=core_y * u.m, z=0 * u.m, frame=tilted_frame)
        grd = project_to_ground(tilt)

        h_max = self.reconstruct_h_max(
            nom.fov_lon,
            nom.fov_lat,
            tilt.x,
            tilt.y,
            hillas_dict_mod,
            tel_x,
            tel_y,
            90 * u.deg - array_pointing.alt,
        )

        src_error = np.sqrt(err_fov_lon**2 + err_fov_lat**2)

        return ReconstructedGeometryContainer(
            alt=sky_pos.altaz.alt.to(u.rad),
            az=sky_pos.altaz.az.to(u.rad),
            core_x=grd.x,
            core_y=grd.y,
            core_tilted_x=tilt.x,
            core_tilted_y=tilt.y,
            core_tilted_uncert_x=u.Quantity(core_err_x, u.m),
            core_tilted_uncert_y=u.Quantity(core_err_y, u.m),
            telescopes=[h for h in hillas_dict_mod.keys()],
            average_intensity=np.mean([h.intensity for h in hillas_dict_mod.values()]),
            is_valid=True,
            alt_uncert=src_error.to(u.rad),
            az_uncert=src_error.to(u.rad),
            h_max=h_max,
            h_max_uncert=u.Quantity(np.nan * h_max.unit),
            goodness_of_fit=np.nan,
            prefix=self.__class__.__name__,
        )

    def reconstruct_nominal(self, hillas_parameters):
        """
        Perform event reconstruction by simple Hillas parameter intersection
        in the nominal system

        Parameters
        ----------
        hillas_parameters: dict
            Hillas parameter objects

        Returns
        -------
        Reconstructed event position in the horizon system

        """
        if len(hillas_parameters) < 2:
            return (np.nan, np.nan, np.nan, np.nan)  # Throw away events with < 2 images

        # Find all pairs of Hillas parameters
        combos = itertools.combinations(list(hillas_parameters.values()), 2)
        hillas_pairs = list(combos)

        # Copy parameters we need to a numpy array to speed things up
        h1 = list(
            map(
                lambda h: [
                    h[0].psi.to_value(u.rad),
                    h[0].fov_lon.to_value(u.rad),
                    h[0].fov_lat.to_value(u.rad),
                    h[0].intensity,
                ],
                hillas_pairs,
            )
        )
        h1 = np.array(h1)
        h1 = np.transpose(h1)

        h2 = list(
            map(
                lambda h: [
                    h[1].psi.to_value(u.rad),
                    h[1].fov_lon.to_value(u.rad),
                    h[1].fov_lat.to_value(u.rad),
                    h[1].intensity,
                ],
                hillas_pairs,
            )
        )
        h2 = np.array(h2)
        h2 = np.transpose(h2)

        # Perform intersection
        sx, sy = self.intersect_lines(h1[1], h1[2], h1[0], h2[1], h2[2], h2[0])

        # Weight by chosen method
        weight = self._weight_method(h1[3], h2[3])
        # And sin of interception angle
        weight *= self.weight_sin(h1[0], h2[0])

        # Make weighted average of all possible pairs
        x_pos = np.average(sx, weights=weight)
        y_pos = np.average(sy, weights=weight)
        var_x = np.average((sx - x_pos) ** 2, weights=weight)
        var_y = np.average((sy - y_pos) ** 2, weights=weight)

        return x_pos, y_pos, np.sqrt(var_x), np.sqrt(var_y)

    def reconstruct_tilted(self, hillas_parameters, tel_x, tel_y):
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

        Returns
        -------
        (float, float, float, float):
            core position X, core position Y, core uncertainty X,
            core uncertainty X
        """
        if len(hillas_parameters) < 2:
            return (np.nan, np.nan, np.nan, np.nan)  # Throw away events with < 2 images

        hill_list = list()
        tx = list()
        ty = list()

        # Need to loop here as dict is unordered
        for tel in hillas_parameters.keys():
            hill_list.append(hillas_parameters[tel])
            tx.append(tel_x[tel])
            ty.append(tel_y[tel])

        # Find all pairs of Hillas parameters
        hillas_pairs = list(itertools.combinations(hill_list, 2))
        tel_x = list(itertools.combinations(tx, 2))
        tel_y = list(itertools.combinations(ty, 2))

        tx = np.zeros((len(tel_x), 2))
        ty = np.zeros((len(tel_y), 2))
        for i, _ in enumerate(tel_x):
            tx[i][0], tx[i][1] = tel_x[i][0].to_value(u.m), tel_x[i][1].to_value(u.m)
            ty[i][0], ty[i][1] = tel_y[i][0].to_value(u.m), tel_y[i][1].to_value(u.m)

        tel_x = np.array(tx)
        tel_y = np.array(ty)

        # Copy parameters we need to a numpy array to speed things up
        hillas1 = map(
            lambda h: [h[0].psi.to_value(u.rad), h[0].intensity], hillas_pairs
        )
        hillas1 = np.array(list(hillas1))
        hillas1 = np.transpose(hillas1)

        hillas2 = map(
            lambda h: [h[1].psi.to_value(u.rad), h[1].intensity], hillas_pairs
        )
        hillas2 = np.array(list(hillas2))
        hillas2 = np.transpose(hillas2)

        # Perform intersection
        crossing_y, crossing_x = self.intersect_lines(
            tel_y[:, 0], tel_x[:, 0], hillas1[0], tel_y[:, 1], tel_x[:, 1], hillas2[0]
        )

        # Weight by chosen method
        weight = self._weight_method(hillas1[1], hillas2[1])
        # And sin of interception angle
        weight *= self.weight_sin(hillas1[0], hillas2[0])

        # Make weighted average of all possible pairs
        x_pos = np.average(crossing_x, weights=weight)
        y_pos = np.average(crossing_y, weights=weight)
        var_x = np.average((crossing_x - x_pos) ** 2, weights=weight)
        var_y = np.average((crossing_y - y_pos) ** 2, weights=weight)

        return x_pos, y_pos, np.sqrt(var_x), np.sqrt(var_y)

    def reconstruct_h_max(
        self, source_x, source_y, core_x, core_y, hillas_parameters, tel_x, tel_y, zen
    ):
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
            Dictionary of telescope X positions in tilted frame
        tel_y: dict
            Dictionary of telescope Y positions in tilted frame
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
            cog_x.append(hillas_parameters[tel].fov_lon.to_value(u.rad))
            cog_y.append(hillas_parameters[tel].fov_lat.to_value(u.rad))
            amp.append(hillas_parameters[tel].intensity)

            tx.append(tel_x[tel].to_value(u.m))
            ty.append(tel_y[tel].to_value(u.m))

        height = get_shower_height(
            source_x.to_value(u.rad),
            source_y.to_value(u.rad),
            np.array(cog_x),
            np.array(cog_y),
            core_x.to_value(u.m),
            core_y.to_value(u.m),
            np.array(tx),
            np.array(ty),
        )
        weight = np.array(amp)
        mean_distance = np.average(height, weights=weight)

        # This value is height above telescope in the tilted system,
        # we should convert to height above ground
        mean_height = mean_distance * np.cos(zen.to_value(u.rad))

        # Add on the height of the detector above sea level
        mean_height += self.subarray.reference_location.geodetic.height.to_value(u.m)

        if mean_height > H_MAX_UPPER_LIMIT_M:
            mean_height = np.nan

        return u.Quantity(mean_height, u.m)

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

        det_ab = a1 * b2 - a2 * b1
        det_bc = b1 * c2 - b2 * c1
        det_ca = c1 * a2 - c2 * a1

        # if  math.fabs(det_ab) < 1e-14 : # /* parallel */
        #    return 0,0
        xs = det_bc / det_ab
        ys = det_ca / det_ab

        return xs, ys

    @staticmethod
    def weight_konrad(p1, p2):
        return (p1 * p2) / (p1 + p2)

    @staticmethod
    def weight_sin(phi1, phi2):
        return np.abs(np.sin(phi1 - phi2))


def get_shower_height(
    source_x, source_y, cog_x, cog_y, core_x, core_y, tel_pos_x, tel_pos_y
):
    """
    Function to calculate the depth of shower maximum geometrically under the assumption
    that the shower maximum lies at the brightest point of the camera image.
    Parameters
    ----------
    source_x: float
        Event source position in nominal frame
    source_y: float
        Event source position in nominal frame
    cog_x: list[float]
        Center of gravity x-position for all the telescopes in rad
    cog_y: list[float]
        Center of gravity y-position for all the telescopes in rad
    core_x: float
        Event core position in telescope tilted frame
    core_y: float
        Event core position in telescope tilted frame
    tel_pos_x: list
        List of telescope X positions in tilted frame
    tel_pos_y: list
        List of telescope Y positions in tilted frame

    Returns
    -------
    float: Depth of maximum of air shower
    """

    # Calculate displacement of image centroid from source position (in rad)
    disp = np.sqrt((cog_x - source_x) ** 2 + (cog_y - source_y) ** 2)
    # Calculate impact parameter of the shower
    impact = np.sqrt((tel_pos_x - core_x) ** 2 + (tel_pos_y - core_y) ** 2)

    # Distance above telescope is ration of these two (small angle)
    height = impact / disp

    return height
