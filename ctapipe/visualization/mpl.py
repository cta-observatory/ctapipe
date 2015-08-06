# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Visualization routines using matplotlib
"""

from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse, RegularPolygon, Rectangle
from numpy import sqrt
import numpy as np
import logging
import copy

__all__ = ['CameraDisplay']

logger = logging.getLogger(__name__)


class CameraDisplay:

    """Camera Display using matplotlib.

    Parameters
    ----------
    geometry : `~ctapipe.io.CameraGeometry`
        Definition of the Camera/Image
    axis : `matplotlib.axes.Axes`
        A matplotlib axes object to plot on, or None to create a new one
    title : str
        Title to put on camera plot
    allow_pick : bool (default False)
        if True, allow user to click and select a pixel
    autoupdate : bool (default True)
        redraw automatically (otherwise need to call plt.draw())
    antialiased : bool  (default True)
        whether to draw in antialiased mode or not.

    Notes
    -----

    Speed: 
        CameraDisplay is not intended to be very fast (matplotlib
        is not a very speed performant graphics library, it is
        intended for nice output plots). However, most of the
        slowness of CameraDisplay is in the constructor.  Once one is
        displayed, changing the image that is displayed is relatively
        fast and efficient. Therefore it is best to initialize an
        instance, and change the data, rather than generating new
        CameraDisplays.

    Pixel Implementation: 
        Pixels are rendered as a
        `matplotlib.collections.PatchCollection` of Polygons (either 6
        or 4 sided).  You can access the PatchCollection directly (to
        e.g. change low-level style parameters) via
        `CameraDisplay.pixels`

    Output: 
        Since CameraDisplay uses matplotlib, any display can be
        saved to any output file supported via
        plt.savefig(filename). This includes `.pdf` and `.png`.

    """

    def __init__(self, geometry, axes=None, title="Camera",
                 allow_pick=False, autoupdate=True, antialiased=True):
        self.axes = axes if axes is not None else plt.gca()
        self.geom = geometry
        self.pixels = None
        self.cmap = plt.cm.jet
        self.autoupdate = autoupdate
        self._active_pixel = None
        self._active_pixel_label = None
        
        # initialize the plot and generate the pixels as a
        # RegularPolyCollection

        patches = []

        for xx, yy, aa in zip(self.geom.pix_x.value,
                              self.geom.pix_y.value,
                              np.array(self.geom.pix_area)):
            if self.geom.pix_type.startswith("hex"):
                rr = sqrt(aa * 2 / 3 / sqrt(3))
                poly = RegularPolygon((xx, yy), 6, radius=rr,
                                      orientation=np.radians(0),
                                      fill=True)
            else:
                rr = sqrt(aa) * sqrt(2)
                poly = Rectangle((xx, yy), width=rr, height=rr,
                                 angle=np.radians(0),
                                 fill=True)

            patches.append(poly)

        self.pixels = PatchCollection(patches, cmap=self.cmap, linewidth=0)
        self.axes.add_collection(self.pixels)
        
        # Set up some nice plot defaults
        
        self.axes.set_aspect('equal', 'datalim')
        self.axes.set_title(title)
        self.axes.set_xlabel("X position ({})".format(self.geom.pix_x.unit))
        self.axes.set_ylabel("Y position ({})".format(self.geom.pix_y.unit))
        self.axes.autoscale_view()

        # set up a patch to display when a pixel is clicked (and
        # pixel_picker is enabled):
        
        self._active_pixel = copy.copy(patches[0])
        self._active_pixel.set_facecolor('r')
        self._active_pixel.set_alpha(0.5)
        self._active_pixel.set_linewidth(2.0)
        self._active_pixel.set_visible(False)
        self.axes.add_patch(self._active_pixel)
        
        self._active_pixel_label = plt.text(self._active_pixel.xy[0],
                                            self._active_pixel.xy[1],
                                            "0",
                                            horizontalalignment='center',
                                            verticalalignment='center')
        self._active_pixel_label.set_visible(False)
        
        # enable ability to click on pixel and do something (can be
        # enabled on-the-fly later as well:
        
        if allow_pick:
            self.enable_pixel_picker()

    def enable_pixel_picker(self):
        """ enable ability to click on pixels """
        self.pixels.set_picker(True)  # enable click
        self.pixels.set_pickradius(sqrt(self.geom.pix_area[0]) / np.pi)
        self.pixels.set_snap(True)  # snap cursor to pixel center
        self.axes.figure.canvas.mpl_connect('pick_event', self._on_pick)

    def set_cmap(self, cmap):
        """ Change the color map 

        Parameters
        ----------
        self: type
            description
        cmap: `matplotlib.colors.ColorMap`
            a color map, e.g. from `matplotlib.pyplot.cm.*`
        """
        self.pixels.set_cmap(cmap)

    def set_image(self, image):
        """
        Change the image displayed on the Camera.

        Parameters
        ----------
        image: array_like
            array of values corresponding to the pixels in the CameraGeometry.
        """
        image = np.asanyarray(image)
        if image.shape != self.geom.pix_x.shape:
            raise ValueError("Image has a different shape {} than the"
                             "given CameraGeometry {}"
                             .format(image.shape, self.geom.pix_x.shape))
        self.pixels.set_array(image)
        self.update()

    def update(self):
        """ signal a redraw if necessary """
        if self.autoupdate:
            plt.draw()

    def add_colorbar(self):
        """ add a colobar to the camera plot """
        self.axes.figure.colorbar(self.pixels)

    def add_ellipse(self, centroid, length, width, angle, asymmetry=0.0,
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
        asymmetry: float
            3rd-order moment for directionality if known
        kwargs:
            any MatPlotLib style arguments to pass to the Ellipse patch

        """
        ellipse = Ellipse(xy=centroid, width=width, height=length,
                          angle=np.degrees(angle), fill=False, **kwargs)
        self.axes.add_patch(ellipse)
        self.update()
        return ellipse

    def overlay_moments(self, momparams, **kwargs):
        """helper to overlay ellipse from a `reco.MomentParameters` structure

        Parameters
        ----------
        momparams: `reco.MomentParameters`
            structuring containing Hillas-style parameterization
        kwargs: key=value
            any style keywords to pass to matplotlib (e.g. color='red'
            or linewidth=6)
        """

        el = self.add_ellipse(centroid=(momparams.cen_x, momparams.cen_y),
                              length=momparams.length,
                              width=momparams.width, angle=momparams.psi,
                              **kwargs)
        self.axes.text(momparams.cen_x, momparams.cen_y,
                       ("({:.02f},{:.02f})\n"
                        "[w={:.02f},l={:.02f}]")
                       .format(momparams.cen_x,
                               momparams.cen_y,
                               momparams.width, momparams.length),
                       color=el.get_edgecolor())

    def _on_pick(self, event):
        """ handler for when a pixel is clicked """
        pix_id = event.ind.pop()
        xx, yy = self.geom.pix_x[pix_id].value, self.geom.pix_y[pix_id].value
        self._active_pixel.xy = (xx, yy)
        self._active_pixel.set_visible(True)
        self._active_pixel_label.set_x(xx)
        self._active_pixel_label.set_y(yy)
        self._active_pixel_label.set_text("{:003d}".format(pix_id))
        self._active_pixel_label.set_visible(True)
        self.update()
        self.on_pixel_clicked(pix_id) # call user-function

    def on_pixel_clicked(self,pix_id):
        """virtual function to overide in sub-classes to do something special
        when a pixel is clicked
        """
        print("Clicked pixel_id {}".format(pix_id))

