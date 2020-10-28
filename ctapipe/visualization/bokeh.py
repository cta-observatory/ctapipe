import sys
from bokeh.events import Tap
import numpy as np
import bokeh
from bokeh.io import output_notebook, push_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource,
    TapTool,
    Span,
    ColorBar,
    LinearColorMapper,
    LogColorMapper,
    ContinuousColorMapper,
    HoverTool,
    BoxZoomTool,
    Ellipse,
    Label,
)
import tempfile
from threading import Timer
from functools import wraps
import astropy.units as u

from ctapipe.instrument import CameraGeometry, PixelShape


PLOTARGS = dict(tools="", toolbar_location=None, outline_line_color="#595959")


# mapper to mpl names
CMAPS = {
    "viridis": bokeh.palettes.Viridis256,
    "magma": bokeh.palettes.Magma256,
    "inferno": bokeh.palettes.Inferno256,
    "grey": bokeh.palettes.Greys256,
    "gray": bokeh.palettes.Greys256,
}


def pallete_from_mpl_name(name):
    if name in CMAPS:
        return CMAPS[name]

    # TODO: make optional if we decide to make one of the plotting
    # TODO: libraries optional
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    rgba = plt.get_cmap(name)(np.linspace(0, 1, 256))
    palette = [to_hex(color) for color in rgba]
    return palette


def is_notebook():
    """
    Returns True if currently running in a notebook session,
    see https://stackoverflow.com/a/37661854/3838691
    """
    return "ipykernel" in sys.modules


def generate_hex_vertices(geom):
    phi = np.arange(0, 2 * np.pi, np.pi / 3)

    # apply pixel rotation and conversion from flat top to pointy top
    phi += geom.pix_rotation.rad + np.deg2rad(30)

    # we need the circumcircle radius, pixel_width is incircle diameter
    r = 2 / np.sqrt(3) * geom.pixel_width.value / 2

    x = geom.pix_x.value
    y = geom.pix_y.value

    xs = x[:, np.newaxis] + r[:, np.newaxis] * np.cos(phi)[np.newaxis]
    ys = y[:, np.newaxis] + r[:, np.newaxis] * np.sin(phi)[np.newaxis]

    return xs, ys


def generate_square_vertices(geom):
    w = geom.pixel_width.value / 2
    x = geom.pix_x.value
    y = geom.pix_y.value

    x_offset = w[:, np.newaxis] * np.array([-1, -1, 1, 1])
    y_offset = w[:, np.newaxis] * np.array([1, -1, -1, 1])

    xs = x[:, np.newaxis] + x_offset
    ys = y[:, np.newaxis] + y_offset
    return xs, ys


class CameraDisplay:
    """
    CameraDisplay implementation in Bokeh
    """

    def __init__(
        # same options as MPL display
        self,
        geometry: CameraGeometry,
        image=None,
        cmap="inferno",
        norm="lin",
        autoscale=True,
        title=None,
        # bokeh specific options
        use_notebook=None,
        autoshow=True,
    ):

        self._geometry = geometry
        self._handle = None
        self._color_bar = None
        self._color_mapper = None
        self._pixels = None
        self._autoshow_timer = None
        # only use autoshow / use_notebook by default if we are in a notebook
        self._use_notebook = use_notebook if use_notebook is not None else is_notebook()

        self._annotations = []
        self._labels = []

        self._init_datasource(image)

        if title is None:
            frame = (
                geometry.frame.__class__.__name__ if geometry.frame else "CameraFrame"
            )
            title = f"{geometry} ({frame})"

        self.figure = figure(title=title, match_aspect=True, aspect_scale=1)

        # Make sure the box zoom tool does not distort the camera display
        for tool in self.figure.toolbar.tools:
            if isinstance(tool, BoxZoomTool):
                tool.match_aspect = True

        # order is important because steps depend on each other
        self.cmap = cmap
        self.norm = norm
        self.autoscale = autoscale
        self.rescale()
        self._setup_camera()

        if autoshow:
            if self._use_notebook:
                self.show()
            else:
                # When running a script, if we would generate a html file
                # directly in __init__, the user would not be able change the display.
                # So give code some time to run before opening the plot,
                # so e.g. colorbars, cmaps and images can be set after the display was created
                self._autoshow_timer = Timer(0.1, self.show)

        if self._autoshow_timer is not None:
            self._autoshow_timer.start()

    def _init_datasource(self, image):
        if image is None:
            image = np.zeros(self._geometry.n_pixels)

        data = dict(id=self._geometry.pix_id, image=image)

        if self._geometry.pix_type == PixelShape.HEXAGON:
            xs, ys = generate_hex_vertices(self._geometry)

        elif self._geometry.pix_type == PixelShape.SQUARE:
            xs, ys = generate_square_vertices(self._geometry)

        elif self._geometry.pix_type == PixelShape.CIRCLE:
            xs, ys = self._geometry.pix_x.value, self._geometry.pix_y.value
            data["radius"] = self._geometry.pixel_width / 2
        else:
            raise NotImplementedError(
                f"Unsupported pixel shape {self._geometry.pix_type}"
            )

        data["xs"], data["ys"] = xs.tolist(), ys.tolist()

        self.datasource = bokeh.plotting.ColumnDataSource(data=data)

    def _reset_autoshow_timer(f):
        """A decorator that resets the timer for autoshow if necessary"""

        @wraps(f)
        def wrapped(self, *args, **kwargs):
            if self._autoshow_timer is not None:
                self._autoshow_timer.cancel()
                self._autoshow_timer = Timer(0.1, self.show)

            res = f(self, *args, **kwargs)

            if self._autoshow_timer is not None:
                self._autoshow_timer.start()

            return res

        return wrapped

    def clear_overlays(self):
        while self._annotations:
            self.figure.renderers.remove(self._annotations.pop())

        while self._labels:
            self.figure.center.remove(self._labels.pop())

    @_reset_autoshow_timer
    def _setup_camera(self):
        kwargs = dict(
            fill_color=dict(field="image", transform=self.norm),
            line_width=0,
            source=self.datasource,
        )
        if self._geometry.pix_type in (PixelShape.SQUARE, PixelShape.HEXAGON):
            self._pixels = self.figure.patches(xs="xs", ys="ys", **kwargs)
        elif self._geometry.pix_type == PixelShape.CIRCLE:
            self._pixels = self.figure.circle(x="xs", y="ys", radius="radius", **kwargs)

        self.figure.add_tools(HoverTool(tooltips=[("id", "@id"), ("value", "@image")]))

    @_reset_autoshow_timer
    def add_colorbar(self):
        self._color_bar = ColorBar(
            color_mapper=self._color_mapper,
            label_standoff=12,
            border_line_color=None,
            location=(0, 0),
        )
        self.figure.add_layout(self._color_bar, "right")
        self.update()

    def update(self):
        if self._use_notebook and self._handle:
            push_notebook(handle=self._handle)

    def rescale(self):
        low = self.datasource.data["image"].min()
        high = self.datasource.data["image"].max()

        # force color to be at lower end of the colormap if
        # data is all equal
        if low == high:
            high += 1

        self.set_limits_minmax(low, high)

    @_reset_autoshow_timer
    def set_limits_minmax(self, zmin, zmax):
        self._color_mapper.update(low=zmin, high=zmax)
        self.update()

    @_reset_autoshow_timer
    def set_limits_percent(self, percent=95):
        zmin = np.nanmin(self.image)
        zmax = np.nanmax(self.image)
        dz = zmax - zmin
        frac = percent / 100.0
        self.set_limits_minmax(zmin, zmax - (1.0 - frac) * dz)

    @property
    def cmap(self):
        return self._palette

    @cmap.setter
    @_reset_autoshow_timer
    def cmap(self, cmap):
        if isinstance(cmap, str):
            cmap = pallete_from_mpl_name(cmap)

        self._palette = cmap
        # might be called in __init__ before color mapper is setup
        if self._color_mapper is not None:
            self._color_mapper.palette = cmap
            self._trigger_cm_update()
            self.update()

    def _trigger_cm_update(self):
        # it seems changing palette does not trigger a color change,
        # so we reassign limits
        low = self._color_mapper.low
        self._color_mapper.update(low=low)

    @property
    def image(self):
        return self.datasource.data["image"]

    @image.setter
    @_reset_autoshow_timer
    def image(self, new_image):
        self.datasource.patch({"image": [(slice(None), new_image)]})
        if self.autoscale:
            self.rescale()

    @property
    def norm(self):
        """
        The norm instance of the Display

        Possible values:

        - "lin": linear scale
        - "log": log scale (cannot have negative values)
        - "symlog": symmetric log scale (negative values are ok)
        """
        return self._color_mapper

    @norm.setter
    @_reset_autoshow_timer
    def norm(self, norm):
        if not isinstance(norm, ContinuousColorMapper):
            if norm == "lin":
                norm = LinearColorMapper
            elif norm == "log":
                norm = LogColorMapper
            else:
                raise ValueError(f"Unsupported norm {norm}")

        self._color_mapper = norm(self.cmap)
        if self._pixels is not None:
            self._pixels.glyph.fill_color.update(transform=self._color_mapper)

        if self._color_bar is not None:
            self._color_bar.update(color_mapper=self._color_mapper)

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
        ellipse = Ellipse(
            x=centroid[0],
            y=centroid[1],
            width=length,
            height=width,
            angle=angle,
            fill_color=None,
            **kwargs,
        )
        glyph = self.figure.add_glyph(ellipse)
        self._annotations.append(glyph)
        self.update()
        return ellipse

    def overlay_moments(
        self, hillas_parameters, with_label=True, keep_old=False, **kwargs
    ):
        """helper to overlay ellipse from a `HillasParametersContainer` structure

        Parameters
        ----------
        hillas_parameters: `HillasParametersContainer`
            structuring containing Hillas-style parameterization
        with_label: bool
            If True, show coordinates of centroid and width and length
        keep_old: bool
            If True, to not remove old overlays
        kwargs: key=value
            any style keywords to pass to matplotlib (e.g. color='red'
            or linewidth=6)
        """
        if not keep_old:
            self.clear_overlays()

        # strip off any units
        cen_x = u.Quantity(hillas_parameters.x).value
        cen_y = u.Quantity(hillas_parameters.y).value
        length = u.Quantity(hillas_parameters.length).value
        width = u.Quantity(hillas_parameters.width).value

        el = self.add_ellipse(
            centroid=(cen_x, cen_y),
            length=length * 2,
            width=width * 2,
            angle=hillas_parameters.psi.to_value(u.rad),
            **kwargs,
        )

        if with_label:
            label = Label(
                x=cen_x,
                y=cen_y,
                text="({:.02f},{:.02f})\n[w={:.02f},l={:.02f}]".format(
                    hillas_parameters.x,
                    hillas_parameters.y,
                    hillas_parameters.width,
                    hillas_parameters.length,
                ),
                text_color=el.line_color,
            )
            self.figure.add_layout(label, "center")
            self._labels.append(label)

    def show(self):
        if self._use_notebook:
            output_notebook()
        else:
            # this only sets the default name, created only when show is called
            output_file(tempfile.mktemp(prefix="ctapipe_bokeh_", suffix=".html"))

        self._handle = show(self.figure, notebook_handle=self._use_notebook)


class WaveformDisplay:
    def __init__(self, waveform=np.zeros(1), fig=None):
        """
        Waveform display that utilises the bokeh visualisation library

        Parameters
        ----------
        waveform : ndarray
            1D array containing the waveform samples
        fig : bokeh.plotting.figure
            Figure to store the bokeh plot onto (optional)
        """
        self._waveform = None
        self._fig = None
        self._active_time = 0

        self.span = None

        cdsource_d = dict(t=[], samples=[])
        self.cdsource = ColumnDataSource(data=cdsource_d)

        self.waveform = waveform
        self.fig = fig

        self.layout = self.fig

    @property
    def fig(self):
        return self._fig

    @fig.setter
    def fig(self, val):
        if val is None:
            val = figure(plot_width=700, plot_height=180, **PLOTARGS)
        self._fig = val

        self._draw_waveform()

    @property
    def waveform(self):
        return self._waveform

    @waveform.setter
    def waveform(self, val):
        if val is None:
            val = np.full(1, np.nan)

        self._waveform = val

        if len(val) == len(self.cdsource.data["t"]):
            self.cdsource.data["samples"] = val
        else:
            cdsource_d = dict(t=np.arange(val.size), samples=val)
            self.cdsource.data = cdsource_d

    @property
    def active_time(self):
        return self._active_time

    @active_time.setter
    def active_time(self, val):
        max_t = self.cdsource.data["t"][-1]
        if val is None:
            val = 0
        if val < 0:
            val = 0
        if val > max_t:
            val = max_t
        self.span.location = val
        self._active_time = val

    def _draw_waveform(self):
        self.fig.line(x="t", y="samples", source=self.cdsource, name="line")

    def enable_time_picker(self):
        """
        Enables the selection of a time by clicking on the waveform
        """
        self.span = Span(
            location=0, dimension="height", line_color="red", line_dash="dashed"
        )
        self.fig.add_layout(self.span)

        taptool = TapTool()
        self.fig.add_tools(taptool)

        def wf_tap_response(event):
            time = event.x
            if time is not None:
                self.active_time = time
                self._on_waveform_click(time)

        self.fig.on_event(Tap, wf_tap_response)

    def _on_waveform_click(self, time):
        print(f"Clicked time: {time}")
        print(f"Active time: {self.active_time}")
