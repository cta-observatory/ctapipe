"""
Line-intersection-based fitting.

Contact: Tino Michael <Tino.Michael@cea.fr>
"""


from ctapipe.reco.reco_algorithms import Reconstructor
from ctapipe.io.containers import ReconstructedShowerContainer
from itertools import combinations

from ctapipe.coordinates import (
    CameraFrame,
    GroundFrame,
    TiltedGroundFrame,
    project_to_ground,
    MissingFrameAttributeWarning,
)
from astropy.coordinates import (
    SkyCoord,
    spherical_to_cartesian,
    cartesian_to_spherical,
    AltAz,
)
import warnings

import numpy as np

from astropy import units as u


__all__ = ['HillasReconstructor', 'TooFewTelescopesException', 'HillasPlane']


class TooFewTelescopesException(Exception):
    pass


def angle(v1, v2):
    """ computes the angle between two vectors
        assuming carthesian coordinates

    Parameters
    ----------
    v1 : numpy array
    v2 : numpy array

    Returns
    -------
    the angle between v1 and v2 as a dimensioned astropy quantity
    """
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.arccos(np.clip(v1.dot(v2) / norm, -1.0, 1.0))


def normalise(vec):
    """ Sets the length of the vector to 1
        without changing its direction

    Parameters
    ----------
    vec : numpy array

    Returns
    -------
    numpy array with the same direction but length of 1
    """
    try:
        return vec / np.linalg.norm(vec)
    except ZeroDivisionError:
        return vec


def line_line_intersection_3d(uvw_vectors, origins):
    '''
    Intersection of many lines in 3d.
    See https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    '''
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


class HillasReconstructor(Reconstructor):
    """
    class that reconstructs the direction of an atmospheric shower
    using a simple hillas parametrisation of the camera images it
    provides a direction estimate in two steps and an estimate for the
    shower's impact position on the ground.

    so far, it does neither provide an energy estimator nor an
    uncertainty on the reconstructed parameters

    """

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        self.hillas_planes = {}

    def predict(self, hillas_dict, inst, pointing_alt, pointing_az):
        '''
        The function you want to call for the reconstruction of the
        event. It takes care of setting up the event and consecutively
        calls the functions for the direction and core position
        reconstruction.  Shower parameters not reconstructed by this
        class are set to np.nan

        Parameters
        -----------
        hillas_dict: dict
            dictionary with telescope IDs as key and
            HillasParametersContainer instances as values
        inst : ctapipe.io.InstrumentContainer
            instrumental description
        pointing_alt: dict[astropy.coordinates.Angle]
            dict mapping telescope ids to pointing altitude
        pointing_az: dict[astropy.coordinates.Angle]
            dict mapping telescope ids to pointing azimuth

        Raises
        ------
        TooFewTelescopesException
            if len(hillas_dict) < 2
        '''

        # filter warnings for missing obs time. this is needed because MC data has no obs time
        warnings.filterwarnings(action='ignore', category=MissingFrameAttributeWarning)

        # stereoscopy needs at least two telescopes
        if len(hillas_dict) < 2:
            raise TooFewTelescopesException(
                "need at least two telescopes, have {}"
                .format(len(hillas_dict)))

        self.initialize_hillas_planes(
            hillas_dict,
            inst.subarray,
            pointing_alt,
            pointing_az
        )

        # algebraic direction estimate
        direction, err_est_dir = self.estimate_direction()

        alt = u.Quantity(list(pointing_alt.values()))
        az = u.Quantity(list(pointing_az.values()))
        if np.any(alt != alt[0]) or np.any(az != az[0]):
            warnings.warn('Divergent pointing not supported')

        telescope_pointing = SkyCoord(alt=alt[0], az=az[0], frame=AltAz())
        # core position estimate using a geometric approach
        core_pos = self.estimate_core_position(hillas_dict, telescope_pointing)

        # container class for reconstructed showers
        result = ReconstructedShowerContainer()
        _, lat, lon = cartesian_to_spherical(*direction)

        # estimate max height of shower
        h_max = self.estimate_h_max()

        # astropy's coordinates system rotates counter-clockwise.
        # Apparently we assume it to be clockwise.
        result.alt, result.az = lat, -lon
        result.core_x = core_pos[0]
        result.core_y = core_pos[1]
        result.core_uncert = np.nan

        result.tel_ids = [h for h in hillas_dict.keys()]
        result.average_intensity = np.mean([h.intensity for h in hillas_dict.values()])
        result.is_valid = True

        result.alt_uncert = err_est_dir
        result.az_uncert = np.nan

        result.h_max = h_max
        result.h_max_uncert = np.nan

        result.goodness_of_fit = np.nan

        return result

    def initialize_hillas_planes(
        self,
        hillas_dict,
        subarray,
        pointing_alt,
        pointing_az
    ):
        """
        creates a dictionary of :class:`.HillasPlane` from a dictionary of
        hillas
        parameters

        Parameters
        ----------
        hillas_dict : dictionary
            dictionary of hillas moments
        subarray : ctapipe.instrument.SubarrayDescription
            subarray information
        tel_phi, tel_theta : dictionaries
            dictionaries of the orientation angles of the telescopes
            needs to contain at least the same keys as in `hillas_dict`
        """
        self.hillas_planes = {}
        horizon_frame = AltAz()
        for tel_id, moments in hillas_dict.items():
            # we just need any point on the main shower axis a bit away from the cog
            p2_x = moments.x + 0.1 * u.m * np.cos(moments.psi)
            p2_y = moments.y + 0.1 * u.m * np.sin(moments.psi)
            focal_length = subarray.tel[tel_id].optics.equivalent_focal_length

            pointing = SkyCoord(
                alt=pointing_alt[tel_id],
                az=pointing_az[tel_id],
                frame=horizon_frame,
            )

            camera_frame = CameraFrame(
                focal_length=focal_length,
                telescope_pointing=pointing
            )

            cog_coord = SkyCoord(
                x=moments.x,
                y=moments.y,
                frame=camera_frame,
            )
            cog_coord = cog_coord.transform_to(horizon_frame)

            p2_coord = SkyCoord(x=p2_x, y=p2_y, frame=camera_frame)
            p2_coord = p2_coord.transform_to(horizon_frame)

            circle = HillasPlane(
                p1=cog_coord,
                p2=p2_coord,
                telescope_position=subarray.positions[tel_id],
                weight=moments.intensity * (moments.length / moments.width),
            )
            self.hillas_planes[tel_id] = circle

    def estimate_direction(self):
        """calculates the origin of the gamma as the weighted average
        direction of the intersections of all hillas planes

        Returns
        -------
        gamma : shape (3) numpy array
            direction of origin of the reconstructed shower as a 3D vector
        crossings : shape (n,3) list
            an error esimate
        """

        crossings = []
        for perm in combinations(self.hillas_planes.values(), 2):
            n1, n2 = perm[0].norm, perm[1].norm
            # cross product automatically weighs in the angle between
            # the two vectors: narrower angles have less impact,
            # perpendicular vectors have the most
            crossing = np.cross(n1, n2)

            # two great circles cross each other twice (one would be
            # the origin, the other one the direction of the gamma) it
            # doesn't matter which we pick but it should at least be
            # consistent: make sure to always take the "upper" solution
            if crossing[2] < 0:
                crossing *= -1
            crossings.append(crossing * perm[0].weight * perm[1].weight)

        result = normalise(np.sum(crossings, axis=0))
        off_angles = [angle(result, cross) for cross in crossings] * u.rad

        err_est_dir = np.average(
            off_angles,
            weights=[len(cross) for cross in crossings]
        )

        return result, err_est_dir

    def estimate_core_position(self, hillas_dict, telescope_pointing):
        '''
        Estimate the core position by intersection the major ellipse lines of each telescope.

        Parameters
        -----------
        hillas_dict: dict[HillasContainer]
            dictionary of hillas moments
        telescope_pointing: SkyCoord[AltAz]
            Pointing direction of the array

        Returns
        -----------
        core_x: u.Quantity
            estimated x position of impact
        core_y: u.Quantity
            estimated y position of impact

        '''
        psi = u.Quantity([h.psi for h in hillas_dict.values()])
        z = np.zeros(len(psi))
        uvw_vectors = np.column_stack([np.cos(psi).value, np.sin(psi).value, z])

        tilted_frame = TiltedGroundFrame(pointing_direction=telescope_pointing)
        ground_frame = GroundFrame()

        positions = [
            (
                SkyCoord(*plane.pos, frame=ground_frame)
                .transform_to(tilted_frame)
                .cartesian.xyz
            )
            for plane in self.hillas_planes.values()
        ]
        core_position = line_line_intersection_3d(uvw_vectors, positions)

        core_pos_tilted = SkyCoord(
            x=core_position[0] * u.m,
            y=core_position[1] * u.m,
            frame=tilted_frame
        )

        core_pos = project_to_ground(core_pos_tilted)

        return core_pos.x, core_pos.y

    def estimate_h_max(self):
        '''
        Estimate the max height by intersecting the lines of the cog directions of each telescope.

        Parameters
        -----------
        hillas_dict : dictionary
            dictionary of hillas moments
        subarray : ctapipe.instrument.SubarrayDescription
            subarray information

        Returns
        -----------
        astropy.unit.Quantity
            the estimated max height
        '''
        uvw_vectors = np.array([plane.a for plane in self.hillas_planes.values()])
        positions = [plane.pos for plane in self.hillas_planes.values()]

        # not sure if its better to return the length of the vector of the z component
        return np.linalg.norm(line_line_intersection_3d(uvw_vectors, positions)) * u.m


class HillasPlane:
    """
    a tiny helper class to collect some parameters for each great great
    circle

    Stores some vectors a, b, and c

    These vectors are eucledian [x, y, z] where positive z values point towards the sky
    and x and y are parallel to the ground.
    """

    def __init__(self, p1, p2, telescope_position, weight=1):
        """The constructor takes two coordinates in the horizontal
        frame (alt, az) which define a plane perpedicular
        to the camera.

        Parameters
        -----------
        p1: astropy.coordinates.SkyCoord
            One of two direction vectors which define the plane.
            This coordinate has to be defined in the ctapipe.coordinates.AltAz
        p2: astropy.coordinates.SkyCoord
            One of two direction vectors which define the plane.
            This coordinate has to be defined in the ctapipe.coordinates.AltAz
        telescope_position: np.array(3)
            Position of the telescope on the ground
        weight : float, optional
            weight of this plane for later use during the reconstruction

        Notes
        -----
        c: numpy.ndarray(3)
            :math:`\vec c = (\vec a \times \vec b) \times \vec a`
            :math:`\rightarrow` a and c form an orthogonal base for the
            great circle
            (only orthonormal if a and b are of unit-length)
        norm: numpy.ndarray(3)
            normal vector of the circle's plane,
            perpendicular to a, b and c
        """

        self.pos = telescope_position

        # astropy's coordinates system rotates counter clockwise. Apparently we assume it to
        # be clockwise
        self.a = np.array(spherical_to_cartesian(1, p1.alt, -p1.az)).ravel()
        self.b = np.array(spherical_to_cartesian(1, p2.alt, -p2.az)).ravel()

        # a and c form an orthogonal basis for the great circle
        # not really necessary since the norm can be calculated
        # with a and b just as well
        self.c = np.cross(np.cross(self.a, self.b), self.a)
        # normal vector for the plane defined by the great circle
        self.norm = normalise(np.cross(self.a, self.c))
        # some weight for this circle
        # (put e.g. uncertainty on the Hillas parameters
        # or number of PE in here)
        self.weight = weight
