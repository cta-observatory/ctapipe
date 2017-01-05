from ctapipe.utils import linalg
from ctapipe.reco.reco_algorithms import RecoShowerGeomAlgorithm
from ctapipe.io.containers import ReconstructedShowerContainer

from astropy.utils.decorators import deprecated

from itertools import combinations, permutations

import numpy as np

from scipy.optimize import minimize

from astropy import units as u
u.dimless = u.dimensionless_unscaled


__all__ = ["FitGammaHillas", "TooFewTelescopesException"]

class TooFewTelescopesException(Exception):
    pass



@deprecated(0.1, "will be replaced with real coord transform")
def guess_pix_direction(pix_x, pix_y, tel_phi, tel_theta, tel_foclen,
                        camera_rotation=0*u.degree):
    '''
        TODO replace with proper implementation
        calculates the direction vector of corresponding to a
        (x,y) position on the camera

        Parameters:
        -----------
        pix_x, pix_y : ndarray
            lists of x and y positions on the camera
        tel_phi, tel_theta: astropy quantities
            two angles that describe the orientation of the telescope
        tel_foclen : astropy quantity
            focal length of the telescope
        camera_rotation : astropy quantity
            rotation of the camera frame inside of the telescope

        Returns:
        --------
        pix_dirs : ndarray
            shape (n,3) list of "direction vectors"
            corresponding to a position on the camera

        Note:
        -----
        beta is the pixel's angular distance to the centre
        according to beta / tel_view = r / maxR
        alpha is the polar angle between the y-axis and the pixel
        to find the direction the pixel is looking at:
        - the pixel direction is set to the telescope direction
        - offset by beta towards up
        - rotated around the telescope direction by the angle alpha
    '''
    pix_alpha = np.arctan2(pix_x, pix_y)
    pix_rho = (pix_x**2 + pix_y**2)**.5

    pix_beta = pix_rho / tel_foclen * u.rad

    tel_dir = linalg.set_phi_theta(tel_phi, tel_theta)

    pix_dirs = []
    for a, b in zip(pix_alpha, pix_beta):
        pix_dir = linalg.set_phi_theta(tel_phi, tel_theta + b)

        pix_dir = linalg.rotate_around_axis(pix_dir, tel_dir,
                                            (a-camera_rotation))
        pix_dirs.append(pix_dir*u.dimless)

    return pix_dirs


def dist_to_traces(core, circles):
    '''
        This function calculates the M-Estimator from the distances of the suggested
        core position to all traces of the given GreatCircles.
        The distance of the core to the trace line is the length of the vector between
        the core and an arbitrary point on the trace projected perpendicular to the trace.

        This is implemented as the scalar product of
        • the connecting vector between the core and the position of the telescope
        • and { trace[1], -trace[0] } as the normal vector of the trace.

        Note:
        -----
        uses the M-Estimator of the distance instead of the distance itself:
        MEst = Σ_i 2*√(1 + d_i²) - 2
    '''

    MEst = 0.
    for circle in circles.values():
        '''
        the distanece of the core '''
        D = core-circle.pos[:2]/u.m
        dist = D[0]*circle.trace[1] - D[1]*circle.trace[0]

        '''
        summing up the M-Estimator with the given circle weights '''
        MEst += (2*np.sqrt(1+(dist**2)) - 2) * circle.weight
    return MEst


def MEst(origin, circles, weights):
    '''
        calculates the M-Estimator:
        a modified χ² that becomes asymptotically linear for high values
        and is therefore less sensitive to outliers

        the test is performed to maximise the angles between the
        fit direction and the all the normal vectors of the great circles

        Parameters:
        -----------
        origin : length-3 array
            direction vector of the gamma's origin used as seed
        circles : GreatCircle array
            collection of great circles created from the camera images
        weights : array
            list of weights for each image/great circle

        Returns:
        --------
        MEstimator : float


        Algorithm:
        ----------
        MEst = Σ_i 2*√(1 + χ²) - 2


        Note:
        -----
        seemingly inferior to negative sum of sin(angle)...
    '''

    sin_ang = np.array([linalg.length(np.cross(origin, circ.norm))
                        for circ in circles.values()])
    return -2*np.sum(weights * np.sqrt((1+np.square(sin_ang)))-2)
    return -np.sum(weights * np.square(sin_ang))

    ang = np.array([linalg.angle(origin, circ.norm)
                    for circ in circles.values()])
    return np.sum(weights*np.sqrt(2. + (ang-np.pi/2.)**2))


def n_angle_sum(origin, circles, weights):
    '''
        calculates the negative sum of the angle between
        the fit direction and all the normal vectors of the great circles

        Parameters:
        -----------
        origin : length-3 array
            direction vector of the gamma's origin used as seed
        circles : GreatCircle array
            collection of great circles created from the camera images
        weights : array
            list of weights for each image/great circle

        Returns:
        --------
        n_sum_angles : float
            negative of the sum of the angles between the test direction
            and all normal vectors of the given great circles
    '''

    sin_ang = np.array([linalg.length(np.cross(origin, circ.norm))
                        for circ in circles.values()])
    return -np.sum(weights*sin_ang)


class FitGammaHillas(RecoShowerGeomAlgorithm):
    '''
        class that reconstructs the direction of an atmospheric shower
        using a simple hillas parametrisation of the camera images
        it provides a direction estimate in two steps
        and an estimate for the shower's impact position on the ground.

        so far, it does neither provide an energy estimator nor
        an uncertainty on the reconstructed parameters
    '''

    def __init__(self):
        # TODO RecoShowerGeomAlgorithm inherits from Component
        # Component needs a parent to be set in __init__
        # super().__init__()
        self.circles = {}

    def predict(self, hillas_dict, inst, tel_phi, tel_theta, seed_pos=(0, 0)):
        '''
            The function you want to call for the reconstruction
            of the event. It takes care of setting up the event
            and consecutively calls the functions for the direction
            and core position reconstruction.
            Shower parameters not reconstructed by this class are
            set to np.nan

            Parameters:
            -----------
            hillas_dict : python dictionary
                dictionary with telescope IDs as key and
                MomentParameters instances as values
            seed_pos : python tuple
                shape (2) tuple with a possible seed for
                the core position fit (e.g. CoG of all telescope images)
        '''

        self.get_great_circles(hillas_dict, inst, tel_phi, tel_theta)
        ''' algebraic direction estimate '''
        dir1 = self.fit_origin_crosses()[0]
        ''' direction estimate using numerical minimisation '''
        # does not really improve the fit for now
        # dir2 = self.fit_origin_minimise(dir1)
        ''' core position estimate using numerical minimisation '''
        pos = self.fit_core(seed_pos)

        ''' container class for reconstructed showers '''
        result = ReconstructedShowerContainer()
        (phi, theta) = linalg.get_phi_theta(dir1)

        # TODO make sure az and phi turn in same direction...
        result.alt, result.az = theta-90*u.deg, phi
        result.core_x = pos[0]
        result.core_y = pos[1]

        result.tel_ids = [h for h in hillas_dict.keys()]
        result.average_size = np.mean([h.size for h in hillas_dict.values()])
        result.is_valid = True

        result.alt_uncert      = np.nan
        result.az_uncert       = np.nan
        result.core_uncert     = np.nan
        result.h_max           = np.nan
        result.h_max_uncert    = np.nan
        result.goodness_of_fit = np.nan

        return result

    def get_great_circles(self, hillas_dict, inst, tel_phi, tel_theta):
        self.circles = {}
        for tel_id, moments in hillas_dict.items():

            p2_x = moments.cen_x + moments.length*np.cos(moments.psi + np.pi/2*u.rad)
            p2_y = moments.cen_y + moments.length*np.sin(moments.psi + np.pi/2*u.rad)

            circle = GreatCircle(
                guess_pix_direction(np.array([moments.cen_x, p2_x])*u.m,
                                    np.array([moments.cen_y, p2_y])*u.m,
                                    tel_phi[tel_id], tel_theta[tel_id],
                                    inst.optical_foclen[tel_id]
                                    ),
                moments.size * (moments.length/moments.width)
                                )
            circle.pos = inst.tel_pos[tel_id]
            self.circles[tel_id] = circle

    def fit_origin_crosses(self):
        ''' calculates the origin of the gamma as the weighted average direction
            of the intersections of all great circles
        '''

        if len(self.circles) < 2:
            raise TooFewTelescopesException(
                "need at least two telescopes, have {}"
                .format(len(self.circles)))

        crossings = []
        for perm in combinations(self.circles.values(), 2):
            n1, n2 = perm[0].norm, perm[1].norm
            ''' cross product automatically weighs in the angle between
                the two vectors:  narrower angles have less impact,
                perpendicular vectors have the most
            '''
            crossing = np.cross(n1, n2)

            ''' two great circles cross each other twice (one would be
                the origin, the other one the direction of the gamma)
                it doesn't matter which we pick but it should at least
                be consistent: make sure to always take the "upper" solution
            '''
            if crossing[2] < 0: crossing *= -1
            crossings.append(crossing * perm[0].weight*perm[1].weight)

        ''' averaging over the solutions of all permutations '''
        return linalg.normalise(np.sum(crossings, axis=0))*u.dimless, crossings

    def fit_origin_minimise(self, seed=[0, 0, 1], test_function=n_angle_sum):
        ''' fits the origin of the gamma with a minimisation procedure
            this function expects that get_great_circles has been run already
            a seed should be given otherwise it defaults to "straight up"
            supperted functions to minimise are an M-estimator and the
            negative sum of the angles to all normal vectors of the
            great circles

            Parameters:
            -----------
            seed : length-3 array
                starting point of the minimisation
            test_function : member function if this class
                either n_angle_sum or MEst (or own implementation...)
                defaults to n_angle_sum if none is given
                n_angle_sum seemingly superior to MEst

            Returns:
            --------
            direction : length-3 numpy array as dimensionless quantity
                best fit for the origin of the gamma from
                the minimisation process
        '''

        '''
            using the sum of the cosines of each direction with
            every otherdirection as weight;
            don't use the product -- with many steep angles, the product will
            become too small and the weight (and the whole fit) useless
        '''
        weights = [np.sum([linalg.length(np.cross(A.norm, B.norm))
                           for A in self.circles.values()]
                          ) * B.weight
                   for B in self.circles.values()]

        ''' minimising the test function '''
        self.fit_result_origin = minimize(test_function, seed,
                                          args=(self.circles, weights),
                                          method='BFGS',
                                          options={'disp': False}
                                          )

        return np.array(linalg.normalise(self.fit_result_origin.x))*u.dimless

    def fit_core(self, seed=[0, 0]*u.m, test_function=dist_to_traces):
        '''
            reconstructs the shower core position from the already set up
            great circles

            Parameters:
            -----------
            seed : python tuple * astropy quantity, optional
                shape (2) tuple with optional starting coordinates in
            test_function : function object, optional
                function to be used by the minimiser
                by default uses self._dist_to_traces

            Note:
            -----
            the core of the shower lies on the cross section of the
            great circle with the horizontal plane the direction of this
            cross section is the cross-product of the normal vectors of
            the circle and the horizontal plane here we only care about
            the direction; not the orientation...
        '''

        '''
        the direction of the cross section of the great circle with the horizontal frame
        is the cross product of the great circle's normal vector with the z-axis '''
        zdir = np.array([0, 0, 1])
        for circle in self.circles.values():
            circle.trace = linalg.normalise(np.cross(circle.norm, zdir))

        # minimising the test function
        self.fit_result_core = minimize(test_function, seed[:2]/u.m,
                                        args=(self.circles),
                                        method='BFGS', options={'disp': False})
        if self.fit_result_core.success:
            return np.array(self.fit_result_core.x) * u.m
        else:
            print("fit no success")
            return np.array(self.fit_result_core.x) * u.m
            return np.array([np.nan]*3) * u.m


class GreatCircle:
    '''
        a tiny helper class to collect some parameters
        for each great great circle
    '''

    def __init__(self, dirs, weight=1):
        ''' the constructor takes two directions on the circle and creates
            the normal vector belonging to that plane

            Parameters:
            -----------
            dirs : shape (2,3) ndarray
                contains two 3D direction-vectors
            weight : float, optional
                weight of this telescope for later use during
                the reconstruction

            Algorithm:
            ----------
            c : length 3 ndarray
                c = (a × b) × a -> a and c form an orthogonal base for the
                great circle (only orthonormal if a and b are of unit-length)
            norm : length 3 ndarray
                normal vector of the circle's plane,
                perpendicular to a, b and c
        '''

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
