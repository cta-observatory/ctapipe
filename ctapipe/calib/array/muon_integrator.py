from shapely.geometry import Polygon,LineString
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import correlate1d
from astropy import units as u
from iminuit import Minuit
from scipy.interpolate import interp2d
__all__ = ['MuonLineIntegrate']


class MuonLineIntegrate(object):
    """
    Object for calculating the expected 2D shape of muon image for a given mirror geometry.
    Geometry is passed to the class as a series of points defining the outer edge of the array
    and a set of points defining the shape of the central hole (if present). Muon profiles are
    then calculated bsed of the line integral along a given axis from the muon impact point.

    Expected 2D images can then be generated when the pixel geometry is passed to the class.
    """
    def __init__(self, mirror_radius, hole_radius, pixel_width=0.2, oversample_bins = 3):
        """
        Class initialisation funtion
        Parameters
        ----------
        mirror_radius: float
            Radius of telescope mirror (circular approx)
        hole_radius: float
            Radius of telescope mirror hole (circular approx)
        pixel_width: float
            width of pixel in camera
        oversample_bins: int
            number of angular bins to evaluate for each pixel width

        Returns
        -------
            None
        """

        self.mirror_radius = mirror_radius
        self.hole_radius = hole_radius
        self.pixel_width = pixel_width
        self.oversample_bins = oversample_bins
        self.pixel_x = 0
        self.pixel_y = 0
        self.image = 0
        self.photemit300_600 = 12165.45

    def chord_length (self,radius,rho,phi):

        sehne = 1 - (rho * rho * np.sin(phi) * np.sin(phi))
        sehne = radius * (np.sqrt(sehne) + rho * np.cos(phi))
        sehne[np.isnan(sehne)] = 0
        sehne[sehne<0] = 0

        return(sehne)

    def intersect_circle(self,r, angle):
        """
        Perform line integration along a given axis in the mirror frame given an impact
        point on the mirrot
        :param impact_x: float
            Impact position on mirror (tilted telescope system)
        :param impact_y: float
            Impact position on mirror (tilted telescope system)
        :param angle: float
            Angle along which to integrate mirror
        :return: float
            length from impact point to mirror edge
        """
        mirror_length = self.chord_length(self.mirror_radius,r/self.mirror_radius,angle)
        hole_length = self.chord_length(self.hole_radius,r/self.hole_radius,angle)
        return mirror_length-hole_length

    def plot_pos(self,impact_dist,radius,phi):
        """
        Perform intersection over all angles and return length
        :param impact_dist: float
            Impact distance from mirror centre
        :param ang: ndarray
            Angles over which to integrate
        :param phi: float
            Rotation angle of muon image
        :return: ndarray
            Chord length for each angle
        """

        bins = int((2 * math.pi * radius)/self.pixel_width.value) * self.oversample_bins
        ang = np.linspace(-1*math.pi+phi, 1*math.pi+phi,bins*1)


        l = self.intersect_circle(impact_dist,ang)
        l = correlate1d(l,np.ones(self.oversample_bins),mode="wrap",axis=0)
        l /= self.oversample_bins

        return ang,l

    def pos_to_angle(self,centre_x,centre_y,pixel_x,pixel_y):
        """
        Convert pixel positions from x,y coordinates to rotation angle
        :param centre_x: float
            Reconstructed image centre
        :param centre_y: float
            Reconstructed image centre
        :param pixel_x: ndarray
            Pixel x position
        :param pixel_y: ndarray
            Pixel y position
        :return: ndarray
            Pixel rotation angle
        """
        del_x = pixel_x - centre_x
        del_y = pixel_y - centre_y

        ang = np.arctan2(del_x,del_y)
        return ang

    def image_prediction(self,impact_dist,phi,centre_x,centre_y,radius,width,pixel_x,pixel_y):
        """
        Function for producing the expected image for a given set of trial muon parameters

        :param impact_dist: float
            Impact distance of muon
        :param centre_x: float
            Muon ring centre in field of view
        :param centre_y: float
            Muon ring centre in field of view
        :param radius: float
            Radius of muon rins
        :param width: float
            Gaussian width of muon ring
        :param pixel_x: ndarray
            Pixel x coordinate
        :param pixel_y: ndarray
            Pixel y coordinate
        :return: ndarray
            Predicted signal
        """
        ang = self.pos_to_angle(centre_x,centre_y,pixel_x,pixel_y)
        ang +=phi
        ang_prof,profile = self.plot_pos(impact_dist,radius,phi)


        #r_bins = np.linspace(-radius*0.4,radius*0.4,radius/self.pixel_width.value)
        #r_gauss = np.exp(-np.power(r_bins,2)/(2*np.power(width,2))) * (1./(2 * np.power(width,2) * math.pi))
        #lookup = (profile[:,np.newaxis]*r_gauss)/np.sum(r_gauss)

        radial_dist = np.sqrt(np.power(pixel_x-centre_x,2) + np.power(pixel_y-centre_y,2))
        ring_dist = radial_dist - radius

        #print(lookup[0],profile[0],r_bins,width)
        pred = np.interp(ang,ang_prof,profile)
        #interp_grid = interp2d(r_bins,ang_prof,lookup,kind="linear")
        #pred2 = interp_grid(ring_dist,ang)

        gauss = np.exp(-np.power(ring_dist,2)/(2*np.power(width,2))) / np.sqrt(2 * math.pi * np.power(width,2))

        # Multiply by integrated emissivity between 300 and 600 nm
        pred *= 0.5 * self.photemit300_600

        pred *= (self.pixel_width / radius)
        pred *= np.sin(2 * radius/57.3)
        pred*=self.pixel_width*gauss

        pixarea_factor = 1
        return pred * pixarea_factor

    def likelihood(self,impact_dist,phi,centre_x,centre_y,radius,width,eff):
        """

        Parameters
        ----------
        impact_dist
        centre_x
        centre_y
        radius
        width
        eff
        Returns
        -------

        """
        prediction = self.image_prediction(impact_dist,phi,centre_x,centre_y,radius,width,self.pixel_x,self.pixel_y)
        prediction *= eff

        return -2 * np.sum(self.calc_likelihood(self.image,prediction,0.5,1.1))

    @staticmethod
    def calc_likelihood(image,pred,spe_width,ped):
        """

        Parameters
        ----------
        image
        pred
        spe_width
        ped

        Returns
        -------

        """
        sq = 1./np.sqrt(2 * math.pi * (np.power(ped ,2)
            + pred*(1+pow(spe_width,2))) )

        diff = np.power(image-pred,2.)
        denom = 2 * (np.power(ped ,2) + pred*(1+pow(spe_width,2)) )
        expo = np.exp(-1 * diff / denom)
        sm = expo<1e-300
        expo[sm] = 1e-300
        return np.log(sq*expo)

    def fit_muon(self,centre_x,centre_y,radius,pixel_x,pixel_y,image):
        """

        Parameters
        ----------
        centre_x
        centre_y
        radius
        pixel_x
        pixel_y
        image

        Returns
        -------

        """
        self.image = image
        self.pixel_x = pixel_x.value
        self.pixel_y = pixel_y.value

        min = Minuit(self.likelihood,impact_dist=4,limit_impact_dist=(0,25),error_impact_dist=5,
                     phi=0,limit_phi=(-1*math.pi,1*math.pi),error_phi=0.1,
                     radius=radius.value,fix_radius=True,centre_x=centre_x.value,fix_centre_x=True,
                     centre_y=centre_y.value,fix_centre_y=True,
                     width=0.1,error_width = 0.001*radius.value,limit_width=(0,1),
                     eff=0.1,error_eff=0.05,limit_eff=(0,1),
                     errordef=1)

        min.migrad()

        fit_params = min.values
        #print(fit_params)
        return fit_params['impact_dist'] ,fit_params['phi'],fit_params['width'],fit_params['eff']
