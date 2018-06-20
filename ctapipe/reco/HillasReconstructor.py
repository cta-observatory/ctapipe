"""
Line-intersection-based fitting.

Contact: Tino Michael <Tino.Michael@cea.fr>
"""


from ctapipe.reco.reco_algorithms import Reconstructor
from ctapipe.io.containers import ReconstructedShowerContainer
from ctapipe.coordinates import TiltedGroundFrame, HorizonFrame, CameraFrame
from astropy.coordinates import SkyCoord, spherical_to_cartesian, cartesian_to_spherical
from itertools import combinations

import numpy as np

from scipy.optimize import minimize

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


class HillasReconstructor(Reconstructor):
    """
    class that reconstructs the direction of an atmospheric shower
    using a simple hillas parametrisation of the camera images it
    provides a direction estimate in two steps and an estimate for the
    shower's impact position on the ground.

    so far, it does neither provide an energy estimator nor an
    uncertainty on the reconstructed parameters

    """

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, parent=tool, **kwargs)
        self.hillas_planes = {}

    def predict(self, hillas_dict, inst, pointing_alt, pointing_az, seed_pos=(0, 0)):
        """The function you want to call for the reconstruction of the
        event. It takes care of setting up the event and consecutively
        calls the functions for the direction and core position
        reconstruction.  Shower parameters not reconstructed by this
        class are set to np.nan

        Parameters
        -----------
        hillas_dict : python dictionary
            dictionary with telescope IDs as key and
            HillasParametersContainer instances as values
        inst : ctapipe.io.InstrumentContainer
            instrumental description
        pointing_alt:
        pointing_az:
        seed_pos : python tuple
            shape (2) tuple with a possible seed for
            the core position fit (e.g. CoG of all telescope images)

        Raises
        ------
        TooFewTelescopesException
            if len(hillas_dict) < 2

        """

        # stereoscopy needs at least two telescopes
        if len(hillas_dict) < 2:
            raise TooFewTelescopesException(
                "need at least two telescopes, have {}"
                .format(len(hillas_dict)))

        self.inititialize_hillas_planes(
            hillas_dict,
            inst.subarray,
            pointing_alt,
            pointing_az
        )

        # algebraic direction estimate
        direction, err_est_dir = self.estimate_direction()

        # core position estimate using a geometric approach
        pos, err_est_pos = self.estimate_core_position()

        # container class for reconstructed showers
        result = ReconstructedShowerContainer()
        _, lat, lon = cartesian_to_spherical(*direction)

        # estimate max height of shower
        h_max = self.estimate_h_max(hillas_dict, inst.subarray, pointing_alt, pointing_az)


        result.alt, result.az = lat, lon
        result.core_x = pos[0]
        result.core_y = pos[1]
        result.core_uncert = err_est_pos

        result.tel_ids = [h for h in hillas_dict.keys()]
        result.average_size = np.mean([h.intensity for h in hillas_dict.values()])
        result.is_valid = True

        result.alt_uncert = err_est_dir
        result.az_uncert = np.nan

        result.h_max = h_max
        result.h_max_uncert = np.nan

        result.goodness_of_fit = np.nan

        return result

    def inititialize_hillas_planes(
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
        for tel_id, moments in hillas_dict.items():
            p2_x = moments.x + 0.1 * u.m * np.cos(moments.psi)
            p2_y = moments.y + 0.1 * u.m * np.sin(moments.psi)
            focal_length = subarray.tel[tel_id].optics.equivalent_focal_length

            pointing = SkyCoord(
                alt=pointing_alt[tel_id],
                az=pointing_az[tel_id],
                frame='altaz'
            )

            hf = HorizonFrame(
                array_direction=pointing,
                pointing_direction=pointing
            )
            cf = CameraFrame(
                focal_length=focal_length,
                array_direction=pointing,
                pointing_direction=pointing
            )

            cog_coord = SkyCoord(x=moments.x, y=moments.y, frame=cf)
            cog_coord = cog_coord.transform_to(hf)

            p2_coord = SkyCoord(x=p2_x, y=p2_y, frame=cf)
            p2_coord = p2_coord.transform_to(hf)

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



    def estimate_core_position(self):
        r"""calculates the core position as the least linear square solution
        of an (over-constrained) equation system

        Notes
        -----
        The basis is the "trace" of each telescope's `HillasPlane` which
        can be determined by the telescope's position P=(Px, Py) and
        the circle's normal vector, projected to the ground n=(nx,
        ny), so that for every r=(x, y) on the trace

        :math:`\vec n \cdot \vec r = \vec n \cdot \vec P` ,

        :math:`n_x \cdot x + n_y \cdot y = d`

        In a perfect world, the traces of all telescopes cross in the
        shower's point of impact. This means that there is one common
        point (x, y) for every telescope, so we can write in matrix
        form:

        .. math::
            :label: fullmatrix

            \begin{pmatrix}
                nx_1  &  ny_1  \\
                \vdots & \vdots \\
                nx_n  &  ny_n
            \end{pmatrix}
                \cdot (x, y) =
            \begin{pmatrix}
                d_1  \\
                \vdots \\
                d_n
            \end{pmatrix}



        or :math:`\boldsymbol{A} \cdot \vec r = \vec D` .

        Since we do not live in a perfect world and there probably is
        no point r that fulfils this equation system, it is solved by
        the method of least linear square:

        .. math::
            :label: rchisqr

            \vec{r}_{\chi^2} = (\boldsymbol{A}^\text{T} \cdot \boldsymbol{A})^{-1}
            \boldsymbol{A}^\text{T} \cdot \vec D


        :math:`\vec{r}_{\chi^2}` minimises the squared difference of


        .. math::

            \vec D - \boldsymbol{A} \cdot \vec r


        Weights are applied to every line of equation :eq:`fullmatrix`
        as stored in circle.weight (assuming they have been set in
        `get_great_circles` or elsewhere).

        Returns
        -------
        r_chisqr: numpy.ndarray(2)
            the minimum :math:`\chi^2` solution for the shower impact position
        pos_uncert: astropy length quantity
            error estimate on the reconstructed core position

        """

        A = np.zeros((len(self.hillas_planes), 2))
        D = np.zeros(len(self.hillas_planes))
        for i, circle in enumerate(self.hillas_planes.values()):
            # apply weight from circle and from the tilt of the circle
            # towards the horizontal plane: simply projecting
            # circle.norm to the ground gives higher weight to planes
            # perpendicular to the ground and less to those that have
            # a steeper angle
            A[i] = circle.weight * circle.norm[:2]
            # since A[i] is used in the dot-product, no need to multiply the
            # weight here
            D[i] = np.dot(A[i], circle.pos[:2])

        # the math from equation (2) would look like this:
        # ATA = np.dot(A.T, A)
        # ATAinv = np.linalg.inv(ATA)
        # ATAinvAT = np.dot(ATAinv, A.T)
        # return np.dot(ATAinvAT, D) * unit

        # instead used directly the numpy implementation
        # speed is the same, just handles already "SingularMatrixError"
        if np.all(np.isfinite(A)) and np.all(np.isfinite(D)):
            # note that NaN values create a value error with MKL
            # installations but not otherwise.
            pos = np.linalg.lstsq(A, D)[0] * u.m
        else:
            return [np.nan, np.nan], [np.nan, np.nan]

        weighted_sum_dist = np.sum([np.dot(pos[:2] - c.pos[:2], c.norm[:2]) * c.weight
                                    for c in self.hillas_planes.values()]) * pos.unit
        norm_sum_dist = np.sum([c.weight * np.linalg.norm(c.norm[:2])
                                for c in self.hillas_planes.values()])
        pos_uncert = abs(weighted_sum_dist / norm_sum_dist)

        return pos, pos_uncert



    def estimate_h_max(self, hillas_dict, subarray, pointing_alt, pointing_az):
        weights = []
        tels = []
        dirs = []

        for tel_id, moments in hillas_dict.items():

            focal_length = subarray.tel[tel_id].optics.equivalent_focal_length

            pointing = SkyCoord(
                alt=pointing_alt[tel_id],
                az=pointing_az[tel_id],
                frame='altaz'
            )

            hf = HorizonFrame(
                array_direction=pointing,
                pointing_direction=pointing
            )
            cf = CameraFrame(
                focal_length=focal_length,
                array_direction=pointing,
                pointing_direction=pointing
            )

            cog_coord = SkyCoord(x=moments.x, y=moments.y, frame=cf)
            cog_coord = cog_coord.transform_to(hf)

            cog_direction = spherical_to_cartesian(1, cog_coord.alt, cog_coord.az)
            cog_direction = np.array(cog_direction).ravel()

            weights.append(self.hillas_planes[tel_id].weight)
            tels.append(self.hillas_planes[tel_id].pos)
            dirs.append(cog_direction)

        # minimising the test function
        pos_max = minimize(dist_to_line3d, np.array([0, 0, 10000]),
                           args=(np.array(tels), np.array(dirs), np.array(weights)),
                           method='BFGS',
                           options={'disp': False}
                           ).x
        return pos_max[2] * u.m


def dist_to_line3d(pos, tels, dirs, weights):
    result = np.average(np.linalg.norm(np.cross((pos - tels), dirs), axis=1),
                        weights=weights)
    return result


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
            This coordinate has to be defined in the ctapipe.coordinates.HorizonFrame
        p2: astropy.coordinates.SkyCoord
            One of two direction vectors which define the plane.
            This coordinate has to be defined in the ctapipe.coordinates.HorizonFrame
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

        self.a = np.array(spherical_to_cartesian(1, p1.alt, p1.az)).ravel()
        self.b = np.array(spherical_to_cartesian(1, p2.alt, p2.az)).ravel()

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
