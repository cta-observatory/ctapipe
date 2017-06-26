""""
Line-intersection-based fitting.

Contact: Tino Michael <Tino.Michael@cea.fr>
"""

from ctapipe.utils import linalg
from ctapipe.reco.reco_algorithms import Reconstructor
from ctapipe.io.containers import ReconstructedShowerContainer

from astropy.utils.decorators import deprecated

from itertools import combinations

import numpy as np

from scipy.optimize import minimize

from astropy import units as u
u.dimless = u.dimensionless_unscaled


__all__ = ['HillasReconstructor',
           'TooFewTelescopesException',
           'dist_to_traces', 'MEst', 'GreatCircle']


class TooFewTelescopesException(Exception):
    pass


@deprecated(0.1, "will be replaced with real coord transform")
def guess_pix_direction(pix_x, pix_y, tel_phi, tel_theta, tel_foclen):
    """
    TODO replace with proper implementation
    calculates the direction vector of corresponding to a
    (x,y) position on the camera

    beta is the pixel's angular distance to the centre
    according to beta / tel_view = r / maxR
    alpha is the polar angle between the y-axis and the pixel
    to find the direction the pixel is looking at:

    - the pixel direction is set to the telescope direction
    - offset by beta towards up
    - rotated around the telescope direction by the angle alpha


    Parameters
    -----------
    pix_x, pix_y : ndarray
        lists of x and y positions on the camera
    tel_phi, tel_theta: astropy quantities
        two angles that describe the orientation of the telescope
    tel_foclen : astropy quantity
        focal length of the telescope

    Returns
    -------
    pix_dirs : ndarray
        shape (n,3) list of "direction vectors"
        corresponding to a position on the camera

    """

    pix_alpha = np.arctan2(pix_y, pix_x)
    pix_rho = (pix_x ** 2 + pix_y ** 2) ** .5

    pix_beta = pix_rho / tel_foclen * u.rad

    tel_dir = linalg.set_phi_theta(tel_phi, tel_theta)

    pix_dirs = []
    for a, b in zip(pix_alpha, pix_beta):
        pix_dir = linalg.set_phi_theta(tel_phi, tel_theta + b)

        pix_dir = linalg.rotate_around_axis(pix_dir, tel_dir, -a)
        pix_dirs.append(pix_dir * u.dimless)

    return pix_dirs


def dist_to_traces(core, circles):
    """This function calculates the M-Estimator from the distances of the
    suggested core position to all traces of the given GreatCircles.
    The distance of the core to the trace line is the length of the
    vector between the core and an arbitrary point on the trace
    projected perpendicular to the trace.

    This is implemented as the scalar product of the connecting vector
    between the core and the position of the telescope and { trace[1],
    -trace[0] } as the normal vector of the trace.

    Notes
    -----
    uses the M-Estimator of the distance instead of the distance itself:

    .. math::

        M_\\text{Est} = \sum_i{ 2 \cdot \sqrt{1 + d_i^2} - 2}


    """

    mest = 0.
    for circle in circles.values():

        # the distanece of the core
        D = core - circle.pos[:2] / u.m
        dist = D[0] * circle.trace[1] - D[1] * circle.trace[0]

        # summing up the M-Estimator with the given circle weights
        mest += (2 * np.sqrt(1 + (dist ** 2)) - 2) * circle.weight
    return mest


def MEst(origin, circles, weights):
    """calculates the M-Estimator: a modified χ² that becomes
    asymptotically linear for high values and is therefore less
    sensitive to outliers

    the test is performed to maximise the angles between the fit
    direction and the all the normal vectors of the great circles

    .. math::

        M_\\text{Est} = \sum_i{ 2 \cdot \sqrt{1 + d_i^2} - 2}


    Notes
    -----
    seemingly inferior to negative sum of sin(angle)...


    Parameters
    -----------
    origin : length-3 array
        direction vector of the gamma's origin used as seed
    circles : GreatCircle array
        collection of great circles created from the camera images
    weights : array
        list of weights for each image/great circle

    Returns
    -------
    MEstimator : float


    """

    sin_ang = np.array([linalg.length(np.cross(origin, circ.norm))
                        for circ in circles.values()])
    return -2 * np.sum(weights * np.sqrt((1 + np.square(sin_ang))) - 2)


def neg_angle_sum(origin, circles, weights):
    """calculates the negative sum of the angle between the fit direction
    and all the normal vectors of the great circles

    Parameters
    -----------
    origin : length-3 array
        direction vector of the gamma's origin used as seed
    circles : GreatCircle array
        collection of great circles created from the camera images
    weights : array
        list of weights for each image/great circle

    Returns
    --------
    n_sum_angles : float
        negative of the sum of the angles between the test direction
        and all normal vectors of the given great circles

    """

    sin_ang = np.array([linalg.length(np.cross(origin, circ.norm))
                        for circ in circles.values()])
    return -np.sum(weights * sin_ang)


class HillasReconstructor(Reconstructor):
    """
    class that reconstructs the direction of an atmospheric shower
    using a simple hillas parametrisation of the camera images it
    provides a direction estimate in two steps and an estimate for the
    shower's impact position on the ground.

    so far, it does neither provide an energy estimator nor an
    uncertainty on the reconstructed parameters

    """

    def __init__(self, configurable=None):
        super().__init__(configurable)
        self.circles = {}

    def predict(self, hillas_dict, inst, tel_phi, tel_theta, seed_pos=(0, 0)):
        """The function you want to call for the reconstruction of the
        event. It takes care of setting up the event and consecutively
        calls the functions for the direction and core position
        reconstruction.  Shower parameters not reconstructed by this
        class are set to np.nan

        Parameters
        -----------
        hillas_dict : python dictionary
            dictionary with telescope IDs as key and
            MomentParameters instances as values
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

        self.get_great_circles(hillas_dict, inst, tel_phi, tel_theta)

        # algebraic direction estimate
        dir = self.fit_origin_crosses()[0]

        # core position estimate using a geometric approach
        pos, pos_uncert = self.fit_core_crosses()

        # numerical minimisations do not really improve the fit
        # direction estimate using numerical minimisation
        # dir = self.fit_origin_minimise(dir)
        #
        # core position estimate using numerical minimisation
        # pos = self.fit_core_minimise(seed_pos)

        # container class for reconstructed showers
        result = ReconstructedShowerContainer()
        phi, theta = linalg.get_phi_theta(dir).to(u.deg)

        # TODO make sure az and phi turn in same direction...
        result.alt, result.az = 90 * u.deg - theta, phi
        result.core_x = pos[0]
        result.core_y = pos[1]
        result.core_uncert = pos_uncert

        result.tel_ids = [h for h in hillas_dict.keys()]
        result.average_size = np.mean([h.size for h in hillas_dict.values()])
        result.is_valid = True

        result.alt_uncert = np.nan
        result.az_uncert = np.nan
        result.h_max = np.nan
        result.h_max_uncert = np.nan
        result.goodness_of_fit = np.nan

        return result

    def get_great_circles(self, hillas_dict, inst, tel_phi, tel_theta):
        """
        creates a dictionary of :class:`.GreatCircle` from a dictionary of hillas
        parameters

        Parameters
        ----------
        hillas_dict : dictionary
            dictionary of hillas moments
        inst : ctapipe instrument container
            the container you get from event.inst
        tel_phi, tel_theta : dictionaries
            dictionaries of the orientation angles of the telescopes
            needs to contain at least the same keys as in `hillas_dict`
        """

        self.circles = {}
        for tel_id, moments in hillas_dict.items():

            p2_x = moments.cen_x + moments.length * np.cos(moments.psi)
            p2_y = moments.cen_y + moments.length * np.sin(moments.psi)

            circle = GreatCircle(
                guess_pix_direction(np.array([moments.cen_x / u.m,
                                              p2_x / u.m]) * u.m,
                                    np.array([moments.cen_y / u.m,
                                              p2_y / u.m]) * u.m,
                                    tel_phi[tel_id], tel_theta[tel_id],
                                    inst.optical_foclen[tel_id]),
                moments.size * (moments.length / moments.width)
            )
            circle.pos = inst.tel_pos[tel_id]
            self.circles[tel_id] = circle

    def fit_origin_crosses(self):
        """calculates the origin of the gamma as the weighted average
        direction of the intersections of all great circles

        Returns
        -------
        gamma : shape (3) numpy array
            direction of origin of the reconstructed shower as a 3D vector
        crossings : shape (n,3) list
            list of all the crossings of the `GreatCircle` list
        """

        crossings = []
        for perm in combinations(self.circles.values(), 2):
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

        # averaging over the solutions of all permutations
        return linalg.normalise(np.sum(crossings, axis=0)) * u.dimless, crossings

    def fit_origin_minimise(self, seed=(0, 0, 1), test_function=neg_angle_sum):
        """ Fits the origin of the gamma with a minimisation procedure this
        function expects that `get_great_circles`
        has been run already. A seed should be given otherwise it defaults to
        "straight up" supperted functions to minimise are an M-estimator and the
        negative sum of the angles to all normal vectors of the great
        circles

        Parameters
        -----------
        seed : length-3 array
            starting point of the minimisation
        test_function : function object, optional (default: neg_angle_sum)
            either neg_angle_sum or MEst (or own implementation...)
            neg_angle_sum seemingly superior to MEst

        Returns
        -------
        direction : length-3 numpy array as dimensionless quantity
            best fit for the origin of the gamma from
            the minimisation process

        """

        # using the sum of the cosines of each direction with every
        # other direction as weight; don't use the product -- with many
        # steep angles, the product will become too small and the
        # weight (and the whole fit) useless
        weights = [np.sum([linalg.length(np.cross(A.norm, B.norm))
                           for A in self.circles.values()]
                          ) * B.weight
                   for B in self.circles.values()]

        # minimising the test function
        self.fit_result_origin = minimize(test_function, seed,
                                          args=(self.circles, weights),
                                          method='BFGS',
                                          options={'disp': False}
                                          )

        return np.array(linalg.normalise(self.fit_result_origin.x)) * u.dimless

    def fit_core_crosses(self, unit=u.m):
        r"""calculates the core position as the least linear square solution
        of an (over-constrained) equation system

        Notes
        -----
        The basis is the "trace" of each telescope's `GreatCircle` which
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

        A = np.zeros((len(self.circles), 2))
        D = np.zeros(len(self.circles))
        for i, circle in enumerate(self.circles.values()):
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
        pos = np.linalg.lstsq(A, D)[0] * u.m

        weighted_sum_dist = np.sum([np.dot(pos[:2] - c.pos[:2], c.norm[:2]) * c.weight
                                    for c in self.circles.values()]) * pos.unit
        norm_sum_dist = np.sum([c.weight * linalg.length(c.norm[:2])
                                for c in self.circles.values()])
        pos_uncert = abs(weighted_sum_dist / norm_sum_dist)

        return pos, pos_uncert

    def fit_core_minimise(self, seed=(0, 0), test_function=dist_to_traces):
        """
        reconstructs the shower core position from the already set up
        great circles

        Notes
        -----
        The core of the shower lies on the cross section of the great
        circle with the horizontal plane. The direction of this cross
        section is the cross-product of the circle's normal vector and
        the horizontal plane.  Here, we only care about the direction;
        not the orientation...


        Parameters
        ----------
        seed : tuple
            shape (2) tuple with optional starting coordinates
            tuple of floats or astropy.length -- if floats, assume metres
        test_function : function object, optional (default: dist_to_traces)
            function to be used by the minimiser

        """

        if type(seed) == u.Quantity:
            unit = seed.unit
        else:
            unit = u.m

        # the direction of the cross section of the great circle with
        # the horizontal frame is the cross product of the great
        # circle's normal vector with the z-axis:
        # n × z = (n[1], -n[0], 0)
        for circle in self.circles.values():
            circle.trace = linalg.normalise(np.array([circle.norm[1],
                                                      -circle.norm[0], 0]))

        # minimising the test function (note: minimize strips seed off its
        # unit)
        self.fit_result_core = minimize(test_function, seed[:2],
                                        args=(self.circles),
                                        method='BFGS', options={'disp': False})

        if not self.fit_result_core.success:
            print("fit_core: fit no success")

        return np.array(self.fit_result_core.x) * unit


class GreatCircle:
    """
    a tiny helper class to collect some parameters for each great great
    circle
    """

    def __init__(self, dirs, weight=1):
        """the constructor takes two directions on the circle and creates the
        normal vector belonging to that plane

        Parameters
        -----------
        dirs : shape (2,3) ndarray
            contains two 3D direction-vectors
        weight : float, optional
            weight of this telescope for later use during the
            reconstruction

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

        self.a = dirs[0]
        self.b = dirs[1]

        # a and c form an orthogonal basis for the great circle
        # not really necessary since the norm can be calculated
        # with a and b just as well
        self.c = np.cross(np.cross(self.a, self.b), self.a)
        # normal vector for the plane defined by the great circle
        self.norm = linalg.normalise(np.cross(self.a, self.c))
        # some weight for this circle
        # (put e.g. uncertainty on the Hillas parameters
        # or number of PE in here)
        self.weight = weight
