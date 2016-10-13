from itertools import combinations,permutations

import numpy as np

from scipy.optimize import minimize

from astropy import units as u
u.dimless = u.dimensionless_unscaled

from ctapipe.utils import linalg

from ctapipe.reco.reco_algorithms import RecoShowerGeomAlgorithm
from ctapipe.io.containers import RecoShowerGeom



__all__ = ["FitGammaHillas"]




def guessPixDirection(pix_x, pix_y, tel_phi, tel_theta, tel_foclen, camera_rotation=-100.893 * u.degree):
    # TODO replace with proper implementation
    # beta is the pixel's angular distance to the centre according to beta / tel_view = r / maxR
    # alpha is the polar angle between the y-axis and the pixel
    # to find the direction the pixel is looking at, 
    #  - the pixel direction is set to the telescope direction
    #  - offset by beta towards up
    #  - rotated around the telescope direction by the angle alpha
    
    pix_alpha = np.arctan2(pix_x, pix_y)
    pix_beta  = (pix_x**2 + pix_y**2)**.5

    pix_beta  = pix_beta / tel_foclen * u.rad 
    
    tel_dir = linalg.set_phi_theta(tel_phi,tel_theta)

    pix_dirs = []
    
    
    for a, b in zip(pix_alpha,pix_beta):
        pix_dir = linalg.set_phi_theta( tel_phi, tel_theta + b )
        
        pix_dir = linalg.rotate_around_axis(pix_dir, tel_dir, (a-camera_rotation))
        pix_dirs.append(pix_dir*u.dimless)
        
    return pix_dirs


class FitGammaHillas(RecoShowerGeomAlgorithm):
    
    def __init__(self):
        self.tel_geom = {}
        self.circles = {}
    
    def setup_geometry(self, telescopes, cameras, optics, phi=180.*u.deg, theta=20.*u.deg):
        self.Ver = 'Feb2016'
        self.TelVer = 'TelescopeTable_Version{}'.format(self.Ver)
        self.CamVer = 'CameraTable_Version{}_TelID'.format(self.Ver)
        self.OptVer = 'OpticsTable_Version{}_TelID'.format(self.Ver)
        
        self.telescopes = telescopes[self.TelVer]
        self.cameras    = lambda tel_id : cameras[self.CamVer+str(tel_id)]
        self.optics     = lambda tel_id : optics [self.OptVer+str(tel_id)]
    
        self.tel_phi   = phi
        self.tel_theta = theta

    def predict(self, Hillas_dict, seed_pos=[0,0] ):
        self.get_great_circles(Hillas_dict)
        dir1 = self.fit_origin_crosses()[0]
        dir2 = self.fit_origin_minimise(dir1)
        
        pos = self.fit_core(seed_pos)
        
        result = RecoShowerGeom("FitGammaHillas")
        (phi, theta) = linalg.get_phi_theta(dir2)
        #TODO make sure az and phi turn in same direction...
        result.alt, result.az = theta-90*u.deg, phi
        result.core_x = pos[0]
        result.core_y = pos[1]
        
        result.tel_ids = [h for h in Hillas_dict.keys()]
        
        result.is_valid = True
        
        result.average_size    = np.mean([ h.size for h in Hillas_dict.values() ])
        
        result.alt_uncert      = -1.
        result.az_uncert       = -1.
        result.core_uncert     = -1.
        result.h_max           = -1.
        result.h_max_uncert    = -1.
        result.goodness_of_fit = -1.
        
        
        
        return result
        
    def get_great_circles(self, Hillas_dict):
        self.circles = {}
        for tel_id, moments in Hillas_dict.items():

            
            camera_rotation = -90.*u.deg
            #if tel_id in TelDict["LST"]          : camera_rotation = -110.893*u.deg

            
            circle = GreatCircle(guessPixDirection( np.array([ moments.cen_x, (moments.cen_x + moments.length * np.cos( moments.psi + np.pi/2 ))] ) * u.m,
                                                    np.array([ moments.cen_y, (moments.cen_y + moments.length * np.sin( moments.psi + np.pi/2 ))] ) * u.m,
                                                    self.tel_phi, self.tel_theta, self.telescopes['FL'][tel_id-1] * u.m, camera_rotation=camera_rotation
                                                  )
                                )
            circle.weight = moments.size * (moments.length/moments.width)
            self.circles[tel_id] = circle
            
    
    def fit_origin_crosses(self):
        """ calculates the origin of the gamma as the weighted average direction
            of the intersections of all great circles
        """
        
        assert len(self.circles) >= 2, "need at least two telescopes, have {}".format(len(self.circles))
        
        crossings = []
        for perm in combinations(self.circles.values(), 2):
            n1,n2 = perm[0].norm, perm[1].norm
            # cross product automatically weighs in the angle between the two vectors
            # narrower angles have less impact, perpendicular vectors have the most
            crossing = np.cross(n1,n2)
            # two great circles cross each other twice
            # (one would be the origin, the other one the direction of the gamma)
            # it doesn't matter which we pick but it should at least be consistent
            # make sure to always take the "upper" solution
            if crossing[2] < 0: crossing *= -1
            crossings.append( crossing*perm[0].weight * perm[0].weight  )
        # averaging over the solutions of all permutations
        return linalg.normalise(sum(crossings))*u.dimless, crossings
            
            

    def fit_origin_minimise(self, seed=[0,0,1], test_function=None):
        """ fits the origin of the gamma with a minimisation procedure
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
                either _n_angle_sum or _MEst (or own implementation...)
                defaults to _n_angle_sum if none is given
                _n_angle_sum seemingly superior to _MEst
            
            Returns:
            --------
            direction : length-3 numpy array as dimensionless quantity
                best fit for the origin of the gamma from the minimisation process
        """
        
        if test_function == None: test_function = self._n_angle_sum
        '''
            using the sum of the cosines of each direction with every other direction
            don't use the product -- with many steep angles, the product will 
            become too small and the weight (and the whole fit) useless
        '''
        weights = [ np.sum( [ linalg.length( np.cross(A.norm,B.norm) ) for A in self.circles.values() ] ) *B.weight 
                    for B in self.circles.values() ]
        ''' minimising the test function '''
        self.fit_result_origin = minimize( test_function, seed, args=(weights),
                                           method='BFGS', options={'disp': False}
                                         )
            
        return np.array(linalg.normalise(self.fit_result_origin.x))*u.dimless
        
    def _MEst(self, origin, weights):
        """ calculates the M-Estimator:
            a modified chi2 that becomes asymptotically linear for high values
            and is therefore less sensitive to outliers
            
            the test is performed to maximise the angles between the fit direction
            and the all the normal vectors of the great circles
            
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
            M-Est = sum[  weight * sqrt( 2 * chi**2 ) ]
            
            
            Note:
            -----
            seemingly inferior to negative sum of angles...
            
        """

        sin_ang = np.array([linalg.length(np.cross(origin,circ.norm)) for circ in self.circles.values()])
        return sum( weights*np.sqrt( 2.+ (sin_ang-np.pi/2.)**2) )
        
        
        ang = np.array([linalg.angle(origin,circ.norm) for circ in self.circles.values()])
        ang[ang>np.pi/2.] = np.pi-ang[ang>np.pi/2]
        return sum( weights*np.sqrt( 2.+ (ang-np.pi/2.)**2) )
    
    def _n_angle_sum(self, origin, weights):
        """ calculates the negative sum of the angle between the fit direction 
            and all the normal vectors of the great circles
            
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
        """
        #sin_ang = np.array([np.dot(origin,circ.norm) for circ in self.circles.values()])
        sin_ang = np.array([linalg.length(np.cross(origin,circ.norm)) for circ in self.circles.values()])
        return -sum(weights*sin_ang)
    
    
    def fit_core(self, seed=[0,0]*u.m, test_function=None):
        if test_function == None: test_function = self._dist_to_traces
        zdir = np.array([0,0,1])
        
        # the core of the shower lies on the cross section of the great circle with the horizontal plane
        # the direction of this cross section is the cross-product of the normal vectors of the circle and the horizontal plane
        # here we only care about the direction; not the orientation...
        for circle in self.circles.values():
            circle.trace = linalg.normalise( np.cross( circle.norm, zdir) )
        
        
        # minimising the test function
        self.fit_result_core = minimize( test_function, seed,
                                         method='BFGS', options={'disp': False}
                                       )
        return np.array(self.fit_result_core.x) * u.m
    
    def _dist_to_traces(self, core):
        sum_dist = 0.
        for tel_id, circle in self.circles.items():
            # the distance of the core to the trace line is the scalar product of 
            # • the connecting vector between the core and a random point on the line 
            #   (e.g. the position of the telescope)
            # • and a normal vector of the trace in the same plane as the trace and the core
            #   (e.g. { trace[1], -trace[0] } )
            D = [core[0]-self.telescopes["TelX"][tel_id-1], core[1]-self.telescopes["TelY"][tel_id-1]]
            sum_dist += np.sqrt( 2 + (D[0]*circle.trace[1] - D[1]*circle.trace[0])**2 / 5 ) * circle.weight
        return sum_dist
    

class GreatCircle:
    """ a tiny helper class to collect some parameters for each great great circle """
    
    def __init__(self, dirs):
        """ the constructor takes two directions on the circle and creates
            the normal vector belonging to that plane
            
            Parameters:
            -----------
            dirs : shape (2,3) narray
                contains two 3D direction-vectors
                
            Algorithm:
            ----------
            c : length 3 narray
                c = (a x b) x a -> a and c form an orthogonal base for the great circle
                (only orthonormal if a and b are of unit-length)
            norm : length 3 narray
                normal vector of the circle's plane, perpendicular to a, b and c
        """
        
        self.a      = dirs[0]
        self.b      = dirs[1]
        
        # a and c form an orthogonal basis for the great circle
        # not really necessary since the norm can be calculated with a and b just as well
        self.c      = np.cross( np.cross(self.a,self.b), self.a ) 
        # normal vector for the plane defined by the great circle
        self.norm   = linalg.normalise( np.cross(self.a,self.c) )
        # some weight for this circle 
        # (put e.g. uncertainty on the Hillas parameters or number of PE in here)
        self.weight = 1.
        
        
