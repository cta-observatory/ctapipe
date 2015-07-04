"""
Visualization routines using MatPlotLib
"""

from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib.collections import RegularPolyCollection
from matplotlib.patches import Ellipse
import numpy as np
import logging

__all__ = ['CameraDisplay']

logger = logging.getLogger(__name__)

class CameraDisplay(object):
    """Camera Display using MatPlotLib

    Parameters
    ----------

    geometry : `~ctapipe.io.camera.CameraGeometry`
         Definition of the Camera/Image

    axis : `matplotlib.axes.AxesSubplot`
         a MatPlotLib Axis object to plot on, or None to create a new one

    Notes
    -----
    
    Implementation detail: Pixels are rendered as a
    `matplotlib.collections.RegularPolyCollection`, which is the most
    efficient way in MatPlotLib to display complex pixel shapes.

    """

    def __init__(self, geometry, axes=None, title="Camera"):
        self.axes = axes if axes is not None else plt.gca()
        self.geom = geometry
        self.pixels = None
        self.cmap = plt.cm.jet

        # initialize the plot and generate the pixels as a
        # RegularPolyCollection

        xx, yy, rr = (self.geom.pix_x.value, self.geom.pix_y.value,
                      self.geom.pix_r)
        offsets = list(zip(xx, yy))

        offset_trans = self.axes.transData
        #offset_trans = transforms.IdentityTransform()
        fig = self.axes.get_figure()
        trans = fig.dpi_scale_trans + transforms.Affine2D().scale(1.0 / 72.0)
        #trans = self.axes.transData
        #trans = transforms.IdentityTransform()
        self.axes.set_aspect('equal', 'datalim')

        if self.geom.pix_type.startswith('hex'):
            self.pixels = RegularPolyCollection(numsides=6,
                                                rotation=np.radians(0),
                                                offsets=offsets,
                                                sizes=self._radius_to_size(rr),
                                                transOffset=offset_trans)
            self.pixels.set_cmap(plt.cm.jet)
            self.pixels.set_linewidth(0)
            self.pixels.set_array(np.zeros_like(self.geom.pix_x))
            #self.pixels.set_transform(trans)
            #self.pixels.set_offset_position('data')
            logger.debug("POS:{}".format(self.pixels.get_offset_position()))
            logger.debug("TRN:{}".format(self.pixels.get_offset_transform()))
            self.axes.add_collection(self.pixels, autolim=True)
        elif self.geom.pix_type.startswith('rect'):
            self.pixels = RegularPolyCollection(numsides=4,
                                                rotation=np.radians(45),
                                                offsets=offsets,
                                                sizes=self._radius_to_size(rr),
                                                transOffset=offset_trans)
            self.pixels.set_cmap(plt.cm.jet)
            self.pixels.set_linewidth(0)
            self.pixels.set_array(np.zeros_like(self.geom.pix_x))
            logger.debug("POS:{}".format(self.pixels.get_offset_position()))
            logger.debug("TRN:{}".format(self.pixels.get_offset_transform()))
            self.axes.add_collection(self.pixels, autolim=True)
        else:
            raise ValueError(
                "Unimplemented pixel type: {}", self.geom.pix_type)


        self.axes.set_title(title)
        self.axes.set_xlabel("X position ({})".format(self.geom.pix_x.unit))
        self.axes.set_ylabel("Y position ({})".format(self.geom.pix_y.unit))
        self.axes.autoscale_view()

    def _radius_to_size(self, radii):
        """compute radius in screen coordinates and returns the size in
        points^2, needed for the size parameter of
        RegularPolyCollection. This may not be needed if the
        transormations are set up correctly

        """
        return radii*np.pi*550  # hard-coded for now until better transform
        #return np.pi * radii ** 2

    def set_cmap(self, cmap):
        """ Change the color map """
        self.pixels.set_cmap(cmap)

    def set_image(self, image):
        """
        Change the image displayed on the Camera. 

        Parameters
        ----------
        image: array_like
            array of values corresponding to the pixels in the CameraGeometry.
        """
        if image.shape != self.geom.pix_x.shape:
            raise ValueError("Image has a different shape {} than the" 
                             "given CameraGeometry {}"
                             .format(image.shape, self.geom.pix_x.shape))
        self.pixels.set_array(image)
        plt.draw()  # is there a better way to update this?

    def add_ellipse(self, centroid, length, width, angle, assymmetry=0.0,
                           **kwargs):
        """
        plot an ellipse on top of the camera

        Parameters
        ----------
        centroid: (float,float)
            position of centroid
        length: float
            major axis
        width: float
            minor axis
        angle: float
            rotation angle wrt "up" about the centroid, clockwise, in radians
        assymmetry: float
            3rd-order moment for directionality if known
        kwargs: 
            any MatPlotLib style arguments to pass to the Ellipse patch

        """
        ellipse = Ellipse(xy=centroid, width=width, height=length,
                          angle=np.degrees(angle), fill=False, **kwargs)
        self.axes.add_patch(ellipse)
        plt.draw()
        return ellipse
