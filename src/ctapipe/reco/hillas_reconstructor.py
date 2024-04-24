"""
Line-intersection-based fitting for reconstruction of direction
and core position of a shower.
"""

import warnings
from itertools import combinations

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, Longitude, SkyCoord, cartesian_to_spherical

from ..containers import CameraHillasParametersContainer, ReconstructedGeometryContainer
from ..coordinates import (
    CameraFrame,
    MissingFrameAttributeWarning,
    TelescopeFrame,
    TiltedGroundFrame,
    altaz_to_righthanded_cartesian,
    project_to_ground,
)
from ..instrument import SubarrayDescription
from .reconstructor import (
    HillasGeometryReconstructor,
    InvalidWidthException,
    TooFewTelescopesException,
)

__all__ = ["HillasReconstructor"]


INVALID = ReconstructedGeometryContainer(
    telescopes=[],
    prefix="HillasReconstructor",
)


def angle(v1, v2):
    """computes the angle between two vectors
        assuming cartesian coordinates

    Parameters
    ----------
    v1 : numpy array
    v2 : numpy array

    Returns
    -------
    the angle between v1 and v2 as a dimensioned astropy quantity
    """
    norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    return np.arccos(np.clip(np.sum(v1 * v2, axis=-1) / norm, -1.0, 1.0))


def normalise(vec):
    """Sets the length of the vector to 1
        without changing its direction

    Parameters
    ----------
    vec : numpy array

    Returns
    -------
    numpy array with the same direction but length of 1
    """
    norm = np.linalg.norm(vec, axis=-1)
    if vec.ndim == 1:
        try:
            return vec / norm
        except ZeroDivisionError:
            return vec

    result = np.zeros(vec.shape)
    mask = norm > 0
    result[mask] = vec[mask] / norm[mask, np.newaxis]
    return result


def line_line_intersection_3d(uvw_vectors, origins):
    """
    Intersection of many lines in 3d.
    See https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    """
    C = []
    S = []
    for n, pos in zip(uvw_vectors, origins):
        n = n.reshape((3, 1))
        norm_matrix = (n @ n.T) - np.eye(3)
        C.append(norm_matrix @ pos)
        S.append(norm_matrix)

    S = np.array(S).sum(axis=0)
    C = np.array(C).sum(axis=0)
    return np.linalg.inv(S) @ C


class HillasReconstructor(HillasGeometryReconstructor):
    """
    class that reconstructs the direction of an atmospheric shower
    using a simple hillas parametrisation of the camera images it
    provides a direction estimate in two steps and an estimate for the
    shower's impact position on the ground.

    so far, it does neither provide an energy estimator nor an
    uncertainty on the reconstructed parameters

    """

    def __init__(
        self, subarray: SubarrayDescription, atmosphere_profile=None, **kwargs
    ):
        super().__init__(
            subarray=subarray, atmosphere_profile=atmosphere_profile, **kwargs
        )
        _cam_radius_m = {
            cam: cam.geometry.guess_radius().to_value(u.m)
            for cam in subarray.camera_types
        }

        telframe = TelescopeFrame()
        _cam_radius_deg = {}
        for cam, radius_m in _cam_radius_m.items():
            point_cam = SkyCoord(0, radius_m, unit=u.m, frame=cam.geometry.frame)
            point_tel = point_cam.transform_to(telframe)
            _cam_radius_deg[cam] = point_tel.fov_lon.to_value(u.deg)

        # store for each tel_id to avoid costly hash of camera
        self._cam_radius_m = {
            tel_id: _cam_radius_m[t.camera] for tel_id, t in subarray.tel.items()
        }
        self._cam_radius_deg = {
            tel_id: _cam_radius_deg[t.camera] for tel_id, t in subarray.tel.items()
        }

    def __call__(self, event):
        """
        Perform the full shower geometry reconstruction.

        Parameters
        ----------
        event : `~ctapipe.containers.ArrayEventContainer`
            The event, needs to have dl1 parameters.
            Will be filled with the corresponding dl2 containers,
            reconstructed stereo geometry and telescope-wise impact position.
        """
        warnings.filterwarnings(action="ignore", category=MissingFrameAttributeWarning)

        try:
            hillas_dict = self._create_hillas_dict(event)
        except (TooFewTelescopesException, InvalidWidthException):
            event.dl2.stereo.geometry[self.__class__.__name__] = INVALID
            self._store_impact_parameter(event)
            return

        # Here we perform some basic quality checks BEFORE applying reconstruction
        # This should be substituted by a DL1 QualityQuery specific to this
        # reconstructor
        (
            tel_ids,
            cog_cartesian,
            p2_cartesian,
            corrected_psi,
            weights,
            telescope_positions,
            array_pointing,
        ) = self.initialize_arrays(event, hillas_dict)

        norm = np.cross(cog_cartesian, p2_cartesian)

        # store core corrected psi values
        for tel_id, psi in zip(tel_ids, corrected_psi):
            event.dl1.tel[tel_id].parameters.core.psi = u.Quantity(
                np.rad2deg(psi), u.deg
            )

        # algebraic direction estimate
        direction, err_est_dir = self.estimate_direction(norm, weights)

        # array pointing is needed to define the tilted frame
        core_pos_ground, core_pos_tilted = self.estimate_core_position(
            array_pointing, corrected_psi, telescope_positions
        )

        # container class for reconstructed showers
        _, lat, lon = cartesian_to_spherical(*direction)

        # estimate max height of shower
        h_max = (
            self.estimate_relative_h_max(cog_cartesian, telescope_positions)
            + self.subarray.reference_location.geodetic.height
        )

        # az is clockwise, lon counter-clockwise, make sure it stays in [0, 2pi)
        az = Longitude(-lon)

        event.dl2.stereo.geometry[
            self.__class__.__name__
        ] = ReconstructedGeometryContainer(
            alt=lat,
            az=az,
            core_x=core_pos_ground.x,
            core_y=core_pos_ground.y,
            core_tilted_x=core_pos_tilted.x,
            core_tilted_y=core_pos_tilted.y,
            telescopes=tel_ids.tolist(),
            average_intensity=np.mean([h.intensity for h in hillas_dict.values()]),
            is_valid=True,
            alt_uncert=err_est_dir,
            az_uncert=err_est_dir,
            h_max=h_max,
            prefix=self.__class__.__name__,
        )

        self._store_impact_parameter(event)

    def initialize_arrays(self, event, hillas_dict):
        """
        Creates flat arrays of needed quantities from the event structure.

        Parameters
        ----------
        hillas_dict : dictionary
            dictionary of hillas moments
        subarray : ctapipe.instrument.SubarrayDescription
            subarray information
        telescopes_pointings: dictionary
            dictionary of pointing direction per each telescope
        array_pointing: SkyCoord[AltAz]
            pointing direction of the array

        Notes
        -----
        The part of the algorithm taking into account divergent pointing mode and
        the correction to the psi angle is explained in :cite:p:`phd-gasparetto`,
        section 7.1.4.
        """
        # get one telescope id to check what frame to use
        altaz = AltAz()

        # Due to tracking the pointing of the array will never be a constant
        array_pointing = SkyCoord(
            az=event.pointing.array_azimuth,
            alt=event.pointing.array_altitude,
            frame=altaz,
        )

        # create arrays / quantities of all needed things for all telescopes
        # so we can do transformations vectorized
        cog1 = np.empty(len(hillas_dict))
        cog2 = np.empty(len(hillas_dict))
        cam_radius = np.empty(len(hillas_dict))
        psi = np.empty(len(hillas_dict))
        weights = np.empty(len(hillas_dict))
        focal_length = np.empty(len(hillas_dict))
        alt = np.empty(len(hillas_dict))
        az = np.empty(len(hillas_dict))
        tel_ids = np.empty(len(hillas_dict), dtype=int)

        hillas_in_camera_frame = False

        for i, (tel_id, hillas) in enumerate(hillas_dict.items()):
            tel_ids[i] = tel_id

            pointing = event.pointing.tel[tel_id]
            alt[i] = pointing.altitude.to_value(u.rad)
            az[i] = pointing.azimuth.to_value(u.rad)

            if isinstance(hillas, CameraHillasParametersContainer):
                hillas_in_camera_frame = True
                cog1[i] = hillas.x.to_value(u.m)
                cog2[i] = hillas.y.to_value(u.m)
                cam_radius[i] = self._cam_radius_m[tel_id]
            else:
                cog1[i] = hillas.fov_lon.to_value(u.deg)
                cog2[i] = hillas.fov_lat.to_value(u.deg)
                cam_radius[i] = self._cam_radius_deg[tel_id]

            psi[i] = hillas.psi.to_value(u.rad)
            weights[i] = hillas.intensity * hillas.length.value / hillas.width.value

            camera = self.subarray.tel[tel_id].camera
            focal_length[i] = camera.geometry.frame.focal_length.to_value(u.m)

        indices = self.subarray.tel_index_array[tel_ids]
        telescope_positions = self.subarray.tel_coords[indices]

        telescope_pointings = SkyCoord(alt=alt, az=az, unit=u.rad, frame=altaz)

        focal_length = u.Quantity(focal_length, u.m, copy=False)
        camera_frame = CameraFrame(
            telescope_pointing=telescope_pointings, focal_length=focal_length
        )

        telescope_frame = TelescopeFrame(telescope_pointing=telescope_pointings)

        p2_1 = cog1 + 0.1 * cam_radius * np.cos(psi)
        p2_2 = cog2 + 0.1 * cam_radius * np.sin(psi)

        if hillas_in_camera_frame:
            cog_coord = SkyCoord(x=cog1, y=cog2, unit=u.m, frame=camera_frame)
            p2_coord = SkyCoord(x=p2_1, y=p2_2, unit=u.m, frame=camera_frame)
        else:
            p2_coord = SkyCoord(
                fov_lon=p2_1, fov_lat=p2_2, unit=u.deg, frame=telescope_frame
            )
            cog_coord = SkyCoord(
                fov_lon=cog1, fov_lat=cog2, unit=u.deg, frame=telescope_frame
            )

        cog_coord = cog_coord.transform_to(altaz)
        p2_coord = p2_coord.transform_to(altaz)
        cog_cart = altaz_to_righthanded_cartesian(alt=cog_coord.alt, az=cog_coord.az)
        p2_cart = altaz_to_righthanded_cartesian(alt=p2_coord.alt, az=p2_coord.az)

        pseudo_nominal = CameraFrame(
            telescope_pointing=array_pointing, focal_length=focal_length
        )
        cog_cam = cog_coord.transform_to(pseudo_nominal)
        p2_cam = p2_coord.transform_to(pseudo_nominal)

        corrected_psi = np.arctan(
            (cog_cam.y.to_value(u.m) - p2_cam.y.to_value(u.m))
            / (cog_cam.x.to_value(u.m) - p2_cam.x.to_value(u.m))
        )

        return (
            tel_ids,
            cog_cart,
            p2_cart,
            corrected_psi,
            weights,
            telescope_positions,
            array_pointing,
        )

    @staticmethod
    def estimate_direction(norm, weight):
        """calculates the origin of the gamma as the weighted average
        direction of the intersections of all hillas planes

        Returns
        -------
        gamma : shape (3) numpy array
            direction of origin of the reconstructed shower as a 3D vector
        crossings : shape (n,3) list
            an error estimate
        """
        index_a, index_b = np.array(list(combinations(range(len(norm)), 2))).T
        crossings = np.cross(norm[index_a], norm[index_b])
        mask = crossings[:, 2] < 0
        crossings[mask] = -crossings[mask]

        result = np.average(
            crossings, weights=weight[index_a] * weight[index_b], axis=0
        )
        result = normalise(result)

        off_angles = angle(result, crossings)
        err_est_dir = np.mean(off_angles)

        return result, u.Quantity(np.rad2deg(err_est_dir), u.deg)

    @staticmethod
    def estimate_core_position(array_pointing, psi, positions):
        """
        Estimate the core position by intersection the major ellipse lines of each telescope.

        Parameters
        ----------
        hillas_dict: dict[HillasContainer]
            dictionary of hillas moments
        array_pointing: SkyCoord[HorizonFrame]
            Pointing direction of the array

        Returns
        -------
        core_x: u.Quantity
            estimated x position of impact
        core_y: u.Quantity
            estimated y position of impact

        Notes
        -----
        The part of the algorithm taking into account divergent pointing mode and
        the usage of a corrected psi angle is explained in :cite:p:`phd-gasparetto`
        section 7.1.4.
        """

        # Since psi has been recalculated in the fake CameraFrame
        # it doesn't need any further corrections because it is now independent
        # of both pointing and cleaning/parametrization frame.
        # This angle will be used to visualize the telescope-wise directions of
        # the shower core the ground.

        # Estimate the position of the shower's core
        # from the TiltedFram to the GroundFrame

        z = np.zeros(len(psi))
        uvw_vectors = np.column_stack([np.cos(psi), np.sin(psi), z])

        tilted_frame = TiltedGroundFrame(pointing_direction=array_pointing)
        positions_tilted = positions.transform_to(tilted_frame).cartesian.xyz.T

        core_position = line_line_intersection_3d(
            uvw_vectors,
            positions_tilted.to_value(u.m),
        )
        core_position = u.Quantity(core_position, u.m, copy=False)
        core_pos_tilted = SkyCoord(
            x=core_position[0],
            y=core_position[1],
            z=u.Quantity(0.0, u.m),
            frame=tilted_frame,
        )

        core_pos_ground = project_to_ground(core_pos_tilted)

        return core_pos_ground, core_pos_tilted

    @staticmethod
    def estimate_relative_h_max(cog_vectors, positions):
        """Estimate the relative (to the observatory) vertical height of
        shower-max by intersecting the lines of the cog directions of each
        telescope.

        Returns
        -------
        astropy.unit.Quantity:
            the estimated height above observatory level (not sea level) of the
            shower-max point

        """
        positions = positions.cartesian.xyz.T.to_value(u.m)
        shower_max = u.Quantity(
            line_line_intersection_3d(cog_vectors, positions),
            u.m,
        )

        return shower_max[2]  # the z-coordinate only
