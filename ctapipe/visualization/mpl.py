# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Visualization routines using matplotlib
"""
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse, RegularPolygon, Rectangle, Circle
from matplotlib.colors import Normalize, LogNorm
from numpy import sqrt
import numpy as np
import logging
import copy
from astropy import units as u

__all__ = ['CameraDisplay', 'ArrayDisplay']

logger = logging.getLogger(__name__)


class CameraDisplay:

    """
    Camera Display using matplotlib.

    Parameters
    ----------
    geometry : `~ctapipe.io.CameraGeometry`
        Definition of the Camera/Image
    image: array_like
        array of values corresponding to the pixels in the CameraGeometry.
    ax : `matplotlib.axes.Axes`
        A matplotlib axes object to plot on, or None to create a new one
    title : str (default "Camera")
        Title to put on camera plot
    norm : str or `matplotlib.color.Normalize` instance (default 'lin')
        Normalization for the color scale.
        Supported str arguments are
        - 'lin': linear scale
        - 'log': logarithmic scale (base 10)
    cmap : str or `matplotlib.colors.Colormap` (default 'hot')
        Color map to use (see `matplotlib.cm`)
    allow_pick : bool (default False)
        if True, allow user to click and select a pixel
    autoupdate : bool (default True)
        redraw automatically (otherwise need to call plt.draw())
    autoscale : bool (default True)
        rescale the vmin/vmax values when the image changes.
        This is set to False if `set_limits_*` is called to explicity
        set data limits.
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

    def __init__(
            self,
            geometry,
            image=None,
            ax=None,
            title="Camera",
            norm="lin",
            cmap="hot",
            allow_pick=False,
            autoupdate=True,
            autoscale=True,
            antialiased=True,
            ):
        self.axes = ax if ax is not None else plt.gca()
        self.geom = geometry
        self.pixels = None
        self.colorbar = None
        self.autoupdate = autoupdate
        self.autoscale = autoscale
        self._active_pixel = None
        self._active_pixel_label = None

        # initialize the plot and generate the pixels as a
        # RegularPolyCollection

        patches = []

        if not hasattr(self.geom, "mask"):
            self.geom.mask = np.ones_like(self.geom.pix_x.value)

        for xx, yy, aa in zip(
            u.Quantity(self.geom.pix_x[self.geom.mask==1]).value,
            u.Quantity(self.geom.pix_y[self.geom.mask==1]).value,
            u.Quantity(np.array(self.geom.pix_area)[self.geom.mask==1]).value):

            if self.geom.pix_type.startswith("hex"):
                rr = sqrt(aa * 2 / 3 / sqrt(3))
                poly = RegularPolygon(
                    (xx, yy), 6, radius=rr,
                    orientation=self.geom.pix_rotation.rad,
                    fill=True,
                )
            else:
                rr = sqrt(aa)
                poly = Rectangle(
                    (xx-rr/2., yy-rr/2.),
                    width=rr,
                    height=rr,
                    angle=self.geom.pix_rotation.deg,
                    fill=True,
                )

            patches.append(poly)

        self.pixels = PatchCollection(patches, cmap=cmap, linewidth=0)
        self.axes.add_collection(self.pixels)

        self.pixel_highlighting = copy.copy(self.pixels)
        self.pixel_highlighting.set_facecolor('none')
        self.pixel_highlighting.set_linewidth(0)
        self.axes.add_collection(self.pixel_highlighting)

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

        self._active_pixel_label = self.axes.text(self._active_pixel.xy[0],
                                                  self._active_pixel.xy[1],
                                                  "0",
                                                  horizontalalignment='center',
                                                  verticalalignment='center')
        self._active_pixel_label.set_visible(False)

        # enable ability to click on pixel and do something (can be
        # enabled on-the-fly later as well:

        if allow_pick:
            self.enable_pixel_picker()

        if image is not None:
            self.image = image
        else:
            self.image = np.zeros_like(self.geom.pix_id, dtype=np.float)

        self.norm = norm

    def highlight_pixels(self, pixels, color='g', linewidth=1, alpha=0.75):
        '''
        Highlight the given pixels with a colored line around them

        Parameters
        ----------
        pixels : index-like
            The pixels to highlight.
            Can either be a list or array of integers or a
            boolean mask of length number of pixels
        color: a matplotlib conform color
            the color for the pixel highlighting
        linewidth: float
            linewidth of the highlighting in points
        alpha: 0 <= alpha <= 1
            The transparency
        '''

        l = np.zeros_like(self.image)
        l[pixels] = linewidth
        self.pixel_highlighting.set_linewidth(l)
        self.pixel_highlighting.set_alpha(alpha)
        self.pixel_highlighting.set_edgecolor(color)
        self.update()

    def enable_pixel_picker(self):
        """ enable ability to click on pixels """
        self.pixels.set_picker(True)  # enable click
        self.pixels.set_pickradius(sqrt(u.Quantity(self.geom.pix_area[0])
                                        .value) / np.pi)
        self.pixels.set_snap(True)  # snap cursor to pixel center
        self.axes.figure.canvas.mpl_connect('pick_event', self._on_pick)

    def set_limits_minmax(self, zmin, zmax):
        """ set the color scale limits from min to max """
        self.pixels.set_clim(zmin, zmax)
        self.autoscale = False
        self.update()

    def set_limits_percent(self, percent=95):
        """ auto-scale the color range to percent of maximum """
        zmin = self.pixels.get_array().min()
        zmax = self.pixels.get_array().max()
        dz = zmax - zmin
        frac = percent / 100.0
        self.autoscale = False
        self.set_limits_minmax(zmin, zmax - (1.0 - frac) * dz)

    @property
    def norm(self):
        '''
        The norm instance of the Display

        Possible values:

        - "lin": linear scale
        - "log": log scale
        -  any matplotlib.colors.Normalize instance, e. g. PowerNorm(gamma=-2)
        '''
        return self.pixels.norm

    @norm.setter
    def norm(self, norm):

        if norm == 'lin':
            self.pixels.norm = Normalize()
        elif norm == 'log':
            self.pixels.norm = LogNorm()
            self.pixels.autoscale()  # this is to handle matplotlib bug #5424
        elif isinstance(norm, Normalize):
            self.pixels.norm = norm
        else:
            raise ValueError('Unsupported norm: {}'.format(norm))

        self.update(force=True)
        self.pixels.autoscale()

    @property
    def cmap(self):
        """
        Color map to use. Either a name or  `matplotlib.colors.ColorMap`
        instance, e.g. from `matplotlib.pyplot.cm`
        """
        return self.pixels.get_cmap()

    @cmap.setter
    def cmap(self, cmap):
        self.pixels.set_cmap(cmap)
        self.update()

    @property
    def image(self):
        """The image displayed on the camera (1D array of pixel values)"""
        return self.pixels.get_array()

    @image.setter
    def image(self, image):
        """
        Change the image displayed on the Camera.

        Parameters
        ----------
        image: array_like
            array of values corresponding to the pixels in the CameraGeometry.
        """
        image = np.asanyarray(image)
        if image.shape != self.geom.pix_x.shape:
            raise ValueError(
                "Image has a different shape {} than the "
                "given CameraGeometry {}"
                .format(image.shape, self.geom.pix_x.shape)
            )

        self.pixels.set_array(image[self.geom.mask==True])
        self.pixels.changed()
        if self.autoscale:
            self.pixels.autoscale()
        self.update()

    def update(self, force=False):
        """ signal a redraw if necessary """
        if self.autoupdate:
            if self.colorbar is not None:
                if force is True:
                    self.colorbar.update_bruteforce(self.pixels)
                else:
                    self.colorbar.update_normal(self.pixels)
                self.colorbar.draw_all()
            self.axes.figure.canvas.draw()

    def add_colorbar(self, **kwargs):
        """
        add a colobar to the camera plot
        kwargs are passed to `figure.colorbar(self.pixels, **kwargs)`
        See matplotlib documentation for the supported kwargs:
        http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure.colorbar
        """
        if self.colorbar is not None:
            raise ValueError(
                'There is already a colorbar attached to this CameraDisplay'
            )
        else:
            self.colorbar = self.axes.figure.colorbar(self.pixels, **kwargs)
        self.update()

    def add_ellipse(self, centroid, length, width, angle, asymmetry=0.0,
                    **kwargs):
        """
        plot an ellipse on top of the camera

        Parameters
        ----------
        centroid: (float, float)
            position of centroid
        length: float
            major axis
        width: float
            minor axis
        angle: float
            rotation angle wrt x-axis about the centroid, anticlockwise, in radians
        asymmetry: float
            3rd-order moment for directionality if known
        kwargs:
            any MatPlotLib style arguments to pass to the Ellipse patch

        """
        ellipse = Ellipse(xy=centroid, width=length, height=width,
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

        # strip off any units
        cen_x = u.Quantity(momparams.cen_x).value
        cen_y = u.Quantity(momparams.cen_y).value
        length = u.Quantity(momparams.length).value
        width = u.Quantity(momparams.width).value


        el = self.add_ellipse(centroid=(cen_x, cen_y),
                              length=length,
                              width=width, angle=momparams.psi.rad,
                              **kwargs)
        self.axes.text(cen_x, cen_y,
                       ("({:.02f},{:.02f})\n"
                        "[w={:.02f},l={:.02f}]")
                       .format(momparams.cen_x,
                               momparams.cen_y,
                               momparams.width, momparams.length),
                       color=el.get_edgecolor())

    def _on_pick(self, event):
        """ handler for when a pixel is clicked """
        pix_id = event.ind[-1]
        xx, yy, aa = u.Quantity(self.geom.pix_x[pix_id]).value, \
                     u.Quantity(self.geom.pix_y[pix_id]).value, \
                     u.Quantity(np.array(self.geom.pix_area)[pix_id])
        if self.geom.pix_type.startswith("hex"):
            self._active_pixel.xy = (xx, yy)
        else:
            rr = sqrt(aa)
            self._active_pixel.xy = (xx - rr / 2., yy - rr / 2.)
        self._active_pixel.set_visible(True)
        self._active_pixel_label.set_x(xx)
        self._active_pixel_label.set_y(yy)
        self._active_pixel_label.set_text("{:003d}".format(pix_id))
        self._active_pixel_label.set_visible(True)
        self.update()
        self.on_pixel_clicked(pix_id)  # call user-function

    def on_pixel_clicked(self, pix_id):
        """virtual function to overide in sub-classes to do something special
        when a pixel is clicked
        """
        print("Clicked pixel_id {}".format(pix_id))

    def show(self):
        self.axes.figure.show()


class ArrayDisplay:

    """
    Display a top-town view of a telescope array
    """

    def __init__(self, telx, tely, tel_type=None, radius=20,
                 axes=None, title="Array", autoupdate=True):

        if tel_type is None:
            tel_type = np.ones(len(telx))
        patches = [Rectangle(xy=(x-radius/2, y-radius/2), width=radius, height=radius, fill=False)
                   for x, y in zip(telx, tely)]

        self.autoupdate = autoupdate
        self.telescopes = PatchCollection(patches, match_original=True)
        self.telescopes.set_clim(1, 9)
        rgb = matplotlib.cm.Set1((tel_type-1)/9)
        self.telescopes.set_edgecolor(rgb)
        self.telescopes.set_linewidth(2.0)

        self.axes = axes if axes is not None else plt.gca()
        self.axes.add_collection(self.telescopes)
        self.axes.set_aspect(1.0)
        self.axes.set_title(title)
        self.axes.set_xlim(-1000, 1000)
        self.axes.set_ylim(-1000, 1000)

        self.axes_hillas = axes if axes is not None else plt.gca()


    @property
    def values(self):
        """An array containing a value per telescope"""
        return self.telescopes.get_array()

    @values.setter
    def values(self, values):
        """ set the telescope colors to display  """
        self.telescopes.set_array(values)
        self._update()

    def _update(self):
        """ signal a redraw if necessary """
        if self.autoupdate:
            plt.draw()

    def add_ellipse(self, centroid, length, width, angle, **kwargs):
        """
        plot an ellipse on top of the camera

        Parameters
        ----------
        centroid: (float, float)
            position of centroid
        length: float
            major axis
        width: float
            minor axis
        angle: float
            rotation angle wrt x-axis about the centroid, anticlockwise, in radians
        asymmetry: float
            3rd-order moment for directionality if known
        kwargs:
            any MatPlotLib style arguments to pass to the Ellipse patch

        """
        ellipse = Ellipse(xy=centroid, width=length, height=width,
                          angle=np.degrees(angle), fill=True,  **kwargs)
        self.axes.add_patch(ellipse)
        return ellipse

    def add_polygon(self, centroid, radius, nsides=3, **kwargs):
        """
        plot a polygon on top of the camera

        Parameters
        ----------
        centroid: (float, float)
            position of centroid
        radius: float
            radius
        nsides: int
            Number of points on polygon
        kwargs:
            any MatPlotLib style arguments to pass to the RegularPolygon patch

        """
        polygon = RegularPolygon(xy=centroid, radius=radius, numVertices=nsides, **kwargs)
        self.axes.add_patch(polygon)
        return polygon

    def overlay_moments(self, momparams, tel_position, scale_fac, **kwargs):
        """helper to overlay ellipse from a `reco.MomentParameters` structure

        Parameters
        ----------
        momparams: `reco.MomentParameters`
            structuring containing Hillas-style parameterization
        tel_position: list
            (x, y) positions of each telescope
        scale_fac: float
            scaling factor to apply to width and length when overlaying moments
        kwargs: key=value
            any style keywords to pass to matplotlib (e.g. color='red'
            or linewidth=6)
        """
        # strip off any units
        ellipse_list = list()
        size_list = list()
        i = 0
        for h in momparams:

            length = u.Quantity(momparams[h].length).value * scale_fac
            width = u.Quantity(momparams[h].width).value * scale_fac
            size_list.append(u.Quantity(momparams[h].size).value)
            tel_x = u.Quantity(tel_position[0][i]).value
            tel_y = u.Quantity(tel_position[1][i]).value
            i += 1

            ellipse = Ellipse(xy=(tel_x,tel_y), width=length, height=width,
                              angle=np.degrees(momparams[h].psi.rad))
            ellipse_list.append(ellipse)

        patches = PatchCollection(ellipse_list, **kwargs)
        patches.set_clim(0, 1000) # Set ellipse colour based on image size
        patches.set_array(np.asarray(size_list))
        self.axes_hillas.add_collection(patches)
