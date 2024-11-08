# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Visualization routines using matplotlib
"""
import copy
import logging

import numpy as np
from astropy import units as u

from ..coordinates import get_representation_component_names
from ..exceptions import OptionalDependencyMissing
from ..instrument import PixelShape
from .utils import build_hillas_overlay

__all__ = ["CameraDisplay"]

logger = logging.getLogger(__name__)


def polar_to_cart(rho, phi):
    """ "returns r, theta(degrees)"""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


class CameraDisplay:
    """
    Camera Display using matplotlib.

    Parameters
    ----------
    geometry : `~ctapipe.instrument.CameraGeometry`
        Definition of the Camera/Image
    image: array_like
        array of values corresponding to the pixels in the CameraGeometry.
    ax : `matplotlib.axes.Axes`
        A matplotlib axes object to plot on, or None to create a new one
    title : str (default "Camera")
        Title to put on camera plot
    norm : str or `matplotlib.colors.Normalize` instance (default 'lin')
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
        This is set to False if ``set_limits_*`` is called to explicitly
        set data limits.

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
        ``CameraDisplay.pixels``

    Output:
        Since CameraDisplay uses matplotlib, any display can be
        saved to any output file supported via
        plt.savefig(filename). This includes ``.pdf`` and ``.png``.

    """

    def __init__(
        self,
        # same options as bokeh display
        geometry,
        image=None,
        cmap="inferno",
        norm="lin",
        autoscale=True,
        title=None,
        # mpl specific options
        allow_pick=False,
        autoupdate=True,
        show_frame=True,
        ax=None,
    ):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import PatchCollection
        except ModuleNotFoundError:
            raise OptionalDependencyMissing("matplotlib") from None

        self.axes = ax if ax is not None else plt.gca()
        self.pixels = None
        self.colorbar = None
        self.autoupdate = autoupdate
        self.autoscale = autoscale
        self._active_pixel = None
        self._active_pixel_label = None
        self._axes_overlays = []

        # derotate camera so we don't duplicate the rotation handling code
        self.geom = copy.deepcopy(geometry)
        self.geom.rotate(self.geom.cam_rotation)
        self.unit = self.geom.pix_x.unit

        if title is None:
            title = geometry.name

        # initialize the plot and generate the pixels as a
        # RegularPolyCollection

        if hasattr(self.geom, "mask"):
            self.mask = self.geom.mask
        else:
            self.mask = np.ones(self.geom.n_pixels, dtype=bool)

        patches = self.create_patches(
            shape=self.geom.pix_type,
            pix_x=self.geom.pix_x.to_value(self.unit)[self.mask],
            pix_y=self.geom.pix_y.to_value(self.unit)[self.mask],
            pix_width=self.geom.pixel_width.to_value(self.unit)[self.mask],
            pix_rotation=self.geom.pix_rotation,
        )
        self.pixels = PatchCollection(patches, cmap=cmap, linewidth=0)

        self.axes.add_collection(self.pixels)

        self.pixel_highlighting = copy.copy(self.pixels)
        self.pixel_highlighting.set_facecolor("none")
        self.pixel_highlighting.set_linewidth(0)
        self.axes.add_collection(self.pixel_highlighting)

        # Set up some nice plot defaults

        self.axes.set_aspect("equal", "datalim")
        self.axes.set_title(title)
        self.axes.autoscale_view()

        if show_frame:
            self.add_frame_name()
        # set up a patch to display when a pixel is clicked (and
        # pixel_picker is enabled):

        self._active_pixel = copy.copy(patches[0])
        self._active_pixel.set_facecolor("r")
        self._active_pixel.set_alpha(0.5)
        self._active_pixel.set_linewidth(2.0)
        self._active_pixel.set_visible(False)
        self.axes.add_patch(self._active_pixel)

        if hasattr(self._active_pixel, "xy"):
            center = self._active_pixel.xy
        else:
            center = self._active_pixel.center

        self._active_pixel_label = self.axes.text(
            *center, "0", horizontalalignment="center", verticalalignment="center"
        )
        self._active_pixel_label.set_visible(False)

        # enable ability to click on pixel and do something (can be
        # enabled on-the-fly later as well:

        if allow_pick:
            self.enable_pixel_picker()

        if image is not None:
            self.image = image
        else:
            self.image = np.zeros_like(self.geom.pix_id, dtype=np.float64)

        self.norm = norm
        self.auto_set_axes_labels()

    @staticmethod
    def create_patches(shape, pix_x, pix_y, pix_width, pix_rotation=0 * u.deg):
        if shape == PixelShape.HEXAGON:
            return CameraDisplay._create_hex_patches(
                pix_x, pix_y, pix_width, pix_rotation
            )

        if shape == PixelShape.CIRCLE:
            return CameraDisplay._create_circle_patches(pix_x, pix_y, pix_width)

        if shape == PixelShape.SQUARE:
            return CameraDisplay._create_square_patches(
                pix_x, pix_y, pix_width, pix_rotation
            )

        raise ValueError(f"Unsupported pixel shape {shape}")

    @staticmethod
    def _create_hex_patches(pix_x, pix_y, pix_width, pix_rotation):
        from matplotlib.patches import RegularPolygon

        orientation = pix_rotation.to_value(u.rad)
        return [
            RegularPolygon(
                (x, y),
                6,
                # convert from incircle to outer circle radius
                radius=w / np.sqrt(3),
                orientation=orientation,
                fill=True,
            )
            for x, y, w in zip(pix_x, pix_y, pix_width)
        ]

    @staticmethod
    def _create_circle_patches(pix_x, pix_y, pix_width):
        from matplotlib.patches import Circle

        return [
            Circle((x, y), radius=w / 2, fill=True)
            for x, y, w in zip(pix_x, pix_y, pix_width)
        ]

    @staticmethod
    def _create_square_patches(pix_x, pix_y, pix_width, pix_rotation):
        from matplotlib.patches import RegularPolygon

        orientation = (pix_rotation + 45 * u.deg).to_value(u.rad)
        return [
            RegularPolygon(
                (x, y),
                4,
                # convert from edge length to outer circle radius
                radius=w / np.sqrt(2),
                orientation=orientation,
                fill=True,
            )
            for x, y, w in zip(pix_x, pix_y, pix_width)
        ]

    def highlight_pixels(self, pixels, color="g", linewidth=1, alpha=0.75):
        """
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
        """

        linewidths = np.zeros_like(self.image)
        linewidths[pixels] = linewidth
        self.pixel_highlighting.set_linewidth(linewidths)
        self.pixel_highlighting.set_alpha(alpha)
        self.pixel_highlighting.set_edgecolor(color)
        self._update()

    def enable_pixel_picker(self):
        """enable ability to click on pixels"""
        self.pixels.set_picker(True)
        self.pixels.set_pickradius(self.geom.pixel_width.to_value(self.unit)[0] / 2)
        self.axes.figure.canvas.mpl_connect("pick_event", self._on_pick)

    def set_limits_minmax(self, zmin, zmax):
        """set the color scale limits from min to max"""
        self.pixels.set_clim(zmin, zmax)
        self._update()

    def set_limits_percent(self, percent=95):
        """auto-scale the color range to percent of maximum"""
        from matplotlib.colors import LogNorm

        zmin = np.nanmin(self.pixels.get_array())
        zmax = np.nanmax(self.pixels.get_array())
        if isinstance(self.pixels.norm, LogNorm):
            zmin = zmin if zmin > 0 else 0.1
            zmax = zmax if zmax > 0 else 0.1

        dz = zmax - zmin
        frac = percent / 100.0
        self.set_limits_minmax(zmin, zmax - (1.0 - frac) * dz)

    @property
    def norm(self):
        """
        The norm instance of the Display

        Possible values:

        - "lin": linear scale
        - "log": log scale (cannot have negative values)
        - "symlog": symmetric log scale (negative values are ok)
        -  any matplotlib.colors.Normalize instance, e. g. PowerNorm(gamma=-2)
        """
        return self.pixels.norm

    @norm.setter
    def norm(self, norm):
        from matplotlib.colors import LogNorm, Normalize, SymLogNorm

        vmin, vmax = self.pixels.norm.vmin, self.pixels.norm.vmax

        if norm == "lin":
            self.pixels.norm = Normalize()
        elif norm == "log":
            vmin = 0.1 if vmin < 0 else vmin
            vmax = 0.2 if vmax < 0 else vmax
            self.pixels.norm = LogNorm(vmin=vmin, vmax=vmax)
            self.pixels.autoscale()  # this is to handle matplotlib bug #5424
        elif norm == "symlog":
            self.pixels.norm = SymLogNorm(linthresh=1.0, base=10, vmin=vmin, vmax=vmax)
            self.pixels.autoscale()
        elif isinstance(norm, Normalize):
            self.pixels.norm = norm
        else:
            raise ValueError(
                "Unsupported norm: '{}', options are 'lin',"
                "'log','symlog', or a matplotlib Normalize object".format(norm)
            )

        self.update()
        self.pixels.autoscale()

    @property
    def cmap(self):
        """
        Color map to use. Either name or `matplotlib.colors.Colormap`
        """
        return self.pixels.get_cmap()

    @cmap.setter
    def cmap(self, cmap):
        self.pixels.set_cmap(cmap)
        self._update()

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
                (
                    "Image has a different shape {} than the " "given CameraGeometry {}"
                ).format(image.shape, self.geom.pix_x.shape)
            )

        self.pixels.set_array(np.ma.masked_invalid(image[self.mask]))
        self.pixels.changed()
        if self.autoscale:
            self.pixels.autoscale()
        self._update()

    def _update(self):
        """signal a redraw if autoupdate is turned on"""
        if self.autoupdate:
            self.update()

    def update(self):
        """redraw the display now"""
        self.axes.figure.canvas.draw()
        if self.colorbar is not None:
            self.colorbar.update_normal(self.pixels)
            self.axes.figure.draw_without_rendering()

    def add_colorbar(self, **kwargs):
        """
        add a colorbar to the camera plot
        kwargs are passed to ``figure.colorbar(self.pixels, **kwargs)``
        See matplotlib documentation for the supported kwargs:
        https://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure.colorbar
        """
        if self.colorbar is not None:
            raise ValueError(
                "There is already a colorbar attached to this CameraDisplay"
            )
        else:
            if "ax" not in kwargs:
                kwargs["ax"] = self.axes
            self.colorbar = self.axes.figure.colorbar(self.pixels, **kwargs)
        self.update()

    def add_ellipse(self, centroid, length, width, angle, asymmetry=0.0, **kwargs):
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
        from matplotlib.patches import Ellipse

        ellipse = Ellipse(
            xy=centroid,
            width=length,
            height=width,
            angle=np.degrees(angle),
            fill=False,
            **kwargs,
        )
        self.axes.add_patch(ellipse)
        self.update()
        return ellipse

    def overlay_coordinate(self, coord, keep_old=False, **kwargs):
        """
        Plot a coordinate into the ``CameraDisplay``

        Parameters
        ----------
        coord : `~astropy.coordinates.SkyCoord`
            The coordinate to plot. Must be able to be transformed into
            the frame of the camera geometry of this display.
            Most of the time, this means you need to add the telescope
            pointing as a coordinate attribute like this:
            ``SkyCoord(..., telescope_pointing=pointing)``
        keep_old : bool
            If False, any previously created overlays will be removed
            before plotting the new one
        kwargs :
            All kwargs are passed to ``matplotlib.Axes.plot``
        """
        if not keep_old:
            self.clear_overlays()

        frame = self.geom.frame
        coord = coord.transform_to(frame)

        x_name, y_name = get_representation_component_names(frame)
        x = getattr(coord, x_name).to_value(self.unit)
        y = getattr(coord, y_name).to_value(self.unit)

        kwargs.setdefault("marker", "*")
        kwargs.setdefault("linestyle", "none")
        (plot,) = self.axes.plot(x, y, **kwargs)
        self._axes_overlays.append(plot)

    def overlay_moments(
        self, hillas_parameters, with_label=True, keep_old=False, n_sigma=1, **kwargs
    ):
        """helper to overlay ellipse from a `~ctapipe.containers.HillasParametersContainer` structure

        Parameters
        ----------
        hillas_parameters: `HillasParametersContainer`
            structuring containing Hillas-style parameterization
        with_label: bool
            If True, show coordinates of centroid and width and length
        keep_old: bool
            If True, to not remove old overlays
        n_sigma: float
            How many sigmas to use for the ellipse
        kwargs: key=value
            any style keywords to pass to matplotlib (e.g. color='red'
            or linewidth=6)
        """
        if not keep_old:
            self.clear_overlays()

        params = build_hillas_overlay(
            hillas_parameters,
            self.unit,
            n_sigma=n_sigma,
            with_label=with_label,
        )

        el = self.add_ellipse(
            centroid=(params["cog_x"], params["cog_y"]),
            length=n_sigma * params["length"] * 2,
            width=n_sigma * params["width"] * 2,
            angle=params["psi_rad"],
            **kwargs,
        )

        self._axes_overlays.append(el)

        if with_label:
            text = self.axes.text(
                params["label_x"],
                params["label_y"],
                params["text"],
                color=el.get_edgecolor(),
                va="bottom",
                ha="center",
                rotation=params["rotation"],
                rotation_mode="anchor",
            )

            self._axes_overlays.append(text)

    def clear_overlays(self):
        """Remove added overlays from the axes"""
        while self._axes_overlays:
            overlay = self._axes_overlays.pop()
            overlay.remove()

    def _on_pick(self, event):
        """handler for when a pixel is clicked"""
        if event.artist is not self.pixels:
            # do nothing if the event was triggered by something
            # other than this displays pixels artist
            return

        pix_id = event.ind[-1]
        x = self.geom.pix_x[pix_id].to_value(self.unit)
        y = self.geom.pix_y[pix_id].to_value(self.unit)

        self._active_pixel.xy = (x, y)
        self._active_pixel.set_visible(True)
        self._active_pixel_label.set_x(x)
        self._active_pixel_label.set_y(y)
        self._active_pixel_label.set_text(f"{pix_id:003d}")
        self._active_pixel_label.set_visible(True)
        self._update()
        self.on_pixel_clicked(pix_id)  # call user-function

    def on_pixel_clicked(self, pix_id):
        """virtual function to override in sub-classes to do something special
        when a pixel is clicked
        """
        print(f"Clicked pixel_id {pix_id}")

    def show(self):
        self.axes.figure.show()

    def auto_set_axes_labels(self):
        """set the axes labels based on the Frame attribute"""
        axes_labels = ("X", "Y")
        if self.geom.frame is not None:
            axes_labels = list(
                self.geom.frame.get_representation_component_names().keys()
            )

        self.axes.set_xlabel(f"{axes_labels[0]}  ({self.geom.pix_x.unit})")
        self.axes.set_ylabel(f"{axes_labels[1]}  ({self.geom.pix_y.unit})")

    def add_frame_name(self, color="grey"):
        """label the frame type of the display (e.g. CameraFrame)"""

        frame_name = (
            self.geom.frame.__class__.__name__
            if self.geom.frame is not None
            else "Unknown Frame"
        )
        self.axes.text(  # position text relative to Axes
            1.0,
            0.0,
            frame_name,
            ha="right",
            va="bottom",
            transform=self.axes.transAxes,
            color=color,
            fontsize="smaller",
        )
