from shapely.geometry import Polygon,LineString
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import correlate1d
from astropy import units as u
from iminuit import Minuit
__all__ = ['MuonLineIntegrate']


class MuonLineIntegrate(object):
    """
    Object for calculating the expected 2D shape of muon image for a given mirror geometry.
    Geometry is passed to the class as a series of points defining the outer edge of the array
    and a set of points defining the shape of the central hole (if present). Muon profiles are
    then calculated bsed of the line integral along a given axis from the muon impact point.

    Expected 2D images can then be generated when the pixel geometry is passed to the class.
    """
    def __init__(self, mirror_points, hole_points, pixel_width=0.2, oversample_bins = 5):
        """
        Initialisation function, store shapes of mirror and hole
        :param mirror_points: list
            List of points defining mirror shape
        :param hole_points: list
            List of points defining hole shape
        :param pixel_width: float
            width of pixel in camera
        :param oversample_bins: int
            number of angular bins to evaluate for each pixel width
        :return: None
        """
        self.mirror = Polygon(mirror_points)
        self.hole = Polygon(hole_points)
        self.pixel_width = pixel_width
        self.oversample_bins = oversample_bins
        self.pixel_x = 0
        self.pixel_y = 0
        self.image = 0

    def intersect_polygon(self,impact_x,impact_y, angle):
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

        #First convert vector on mirror to a line
        x1,y1 = self.dir_to_line(impact_x,impact_y, angle,length=100*u.m)
        line = [(impact_x.value, impact_y.value), (x1.value,y1.value)]

        #Create shapely line
        shapely_line = LineString(line)
        try:
            #intersect with polygon
            intersection_line = list(self.mirror.intersection(shapely_line).coords)
        except:
            # incase no interception found return 0
            return 0
        line_intersect = LineString(intersection_line)

        # then do the same for the hole
        try:
            intersection_hole = list(self.hole.intersection(shapely_line).coords)
            hole_intersect = LineString(intersection_hole)
            length_hole = hole_intersect.length
        except:
            length_hole = 0

        #line integral is the difference of these two
        return line_intersect.length - length_hole

    def dir_to_line(self,centre_x,centre_y,angle, length=500*u.m):
        """
        Convert vector style definition of line (point + angle) to two points needed
        for shapely lines
        :param centre_x: float
            Impact point
        :param centre_y: float
            Impact point
        :param angle: float
            Direction of line
        :param length: float
            Length of line
        :return: float,float
            End point of line
        """
        del_x = length * np.sin(angle)
        del_y = length * np.cos(angle)
        x1 = centre_x + del_x
        y1 = centre_y + del_y

        return x1,y1

    def plot_pos(self,impact_x,impact_y,radius):
        """
        Perform intersection over all angles and return length
        :param impact_x: float
            Impact point on mirror
        :param impact_y: float
            Impact point on mirror
        :param ang: ndarray
            Angles over which to integrate
        :return: ndarray
            Chord length for each angle
        """

        #Currently this is don't with a loop, should be made more pythonic later!
        bins = int((2 * math.pi * radius)/self.pixel_width) * self.oversample_bins
        ang = np.linspace(-2*math.pi, 2*math.pi,bins*2)
        l = np.zeros(bins*2)

        i=0
        for a in ang:
            l[i] = self.intersect_polygon(impact_x,impact_y,a)
            i += 1
        l = correlate1d(l,np.ones(self.oversample_bins),mode="wrap")
        l /= self.oversample_bins

        return ang,l

    def pos_to_angle(self,centre_x,centre_y,pixel_x,pixel_y):
        """

        :param centre_x:
        :param centre_y:
        :param pixel_x:
        :param pixel_y:
        :return:
        """
        del_x = pixel_x - centre_x
        del_y = pixel_y - centre_y

        ang = np.arctan2(del_x,del_y)
        return ang

    def image_prediction(self,impact_x,impact_y,centre_x,centre_y,radius,width,pixel_x,pixel_y):
        """

        :param impact_x:
        :param impact_y:
        :param centre_x:
        :param centre_y:
        :param radius:
        :param width:
        :param pixel_x:
        :param pixel_y:
        :return:
        """
        ang = self.pos_to_angle(centre_x,centre_y,pixel_x,pixel_y)

        #creat
        #pred = np.zeros(pixel_x.shape)
        ang_prof,profile = self.plot_pos(impact_x,impact_y,radius)
        pred = np.interp(ang,ang_prof,profile)
        print("here")

        radial_dist = np.sqrt(np.power(pixel_x-centre_x,2) + np.power(pixel_y-centre_y,2))
        ring_dist = radial_dist - radius
        print("here2")

        pred = pred * np.exp(-np.power(ring_dist,2)/(2*np.power(width,2))) * (1./(2 * np.power(width.value,2) * math.pi))
        print("here3")

        return pred

    def likelihood(self,impact_x,impact_y,centre_x,centre_y,radius,width):
        """

        Parameters
        ----------
        impact_x
        impact_y
        centre_x
        centre_y
        radius
        width

        Returns
        -------

        """
        prediction = self.image_prediction(impact_x,impact_y,centre_x,centre_y,radius,width,self.pixel_x,self.pixel_y)
        return np.sum(np.abs(prediction-self.image))

    def fit_muon(self,centre_x,centre_y,radius,pixel_x,pixel_y,image):
        self.image = image
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y

        min = Minuit(self.likelihood,impact_x=0*u.m,error_impact_x=1*u.m,impact_y=0*u.m,error_impact_y=1*u.m,
                     radius=radius,error_radius=radius*0.1,centre_x=centre_x,error_centre_x=centre_x*0.1,
                     centre_y=centre_y,error_centre_y=0.1*centre_y,
                     width=0.01*radius,error_width = 0.001*radius,errordef=1)

        min.migrad()