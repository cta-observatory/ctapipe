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

    Notes
    -----
    Implementation detail: Pixels are rendered as a
    `matplotlib.collections.RegularPolyCollection`, which is the most
    efficient way in matplotlib to display complex pixel shapes.
    """

    def __init__(self, geometry, axes=None, title="Camera",
                 allow_pick=False, autoupdate=True):
        self.axes = axes if axes is not None else plt.gca()
        self.geom = geometry
        self.pixels = None
        self.cmap = plt.cm.jet
        self.autoupdate = autoupdate
        self._active_pixel_id = None
        self._active_pixel = None

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

        self._active_pixel = copy.copy(patches[0])
        self._active_pixel.set_facecolor('r')
        self._active_pixel.set_alpha(0.5)
        self._active_pixel.set_linewidth(2.0)
        self._active_pixel.set_visible(False)

        self.axes.add_collection(self.pixels)
        self.axes.add_patch(self._active_pixel)
        self.axes.set_aspect('equal', 'datalim')
        self.axes.set_title(title)
        self.axes.set_xlabel("X position ({})".format(self.geom.pix_x.unit))
        self.axes.set_ylabel("Y position ({})".format(self.geom.pix_y.unit))
        self.axes.autoscale_view()

        # enable ability to click on pixel and do something
        if allow_pick:
            self.pixels.set_picker(True)  # enable clik
            self.pixels.set_pickradius(sqrt(self.geom.pix_area[0]) / np.pi)
            self.pixels.set_snap(True)  # snap cursor to pixel center
            self.axes.figure.canvas.mpl_connect('pick_event', self._on_pick)

    def _radius_to_size(self, radii):
        """compute radius in screen coordinates and returns the size in
        points^2, needed for the size parameter of
        RegularPolyCollection. This may not be needed if the
        transormations are set up correctly

        """
        return radii * np.pi * 550  # hard-coded for now until better transform
        # return np.pi * radii ** 2

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
        self: type
            description
        momparams: `reco.MomentParameters`
            structuring containing Hillas-style parameterization

        """

        self.add_ellipse(centroid=(momparams.cen_x, momparams.cen_y),
                         length=momparams.length,
                         width=momparams.width, angle=momparams.psi,
                         **kwargs)
        self.axes.text( momparams.cen_x, momparams.cen_y,
                        ("({:.02f},{:.02f})\n"
                         "[w={:.02f},l={:.02f}]")
                        .format(momparams.cen_x,
                                momparams.cen_y,
                                momparams.width, momparams.length ))

    def _on_pick(self, event):
        """ handler for when a pixel is clicked """
        print("Clicked pixel_id {}".format(event.ind))
        pix_id = event.ind.pop()
        self._active_pixel.set_visible(True)
        self._active_pixel.xy = (self.geom.pix_x[pix_id].value,
                                 self.geom.pix_y[pix_id].value)
        self.update()
