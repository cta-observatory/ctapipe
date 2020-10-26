import sys
from bokeh.events import Tap
import numpy as np
import bokeh
from bokeh.io import output_notebook, push_notebook, show
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource,
    TapTool,
    Span,
    ColorBar,
    LinearColorMapper,
    HoverTool,
    BoxZoomTool,
)

from ctapipe.instrument import CameraGeometry, PixelShape


PLOTARGS = dict(tools="", toolbar_location=None, outline_line_color="#595959")


# mapper to mpl names
CMAPS = {
    "viridis": bokeh.palettes.Viridis256,
    "magma": bokeh.palettes.Magma256,
    "inferno": bokeh.palettes.Inferno256,
    "grey": bokeh.palettes.Greys256,
}


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


class CameraDisplay:
    """
    CameraDisplay implementation in Bokeh
    """

    def __init__(
        self, geom: CameraGeometry, image=None, use_notebook=None, autoshow=True
    ):

        self._geom = geom
        self._handle = None

        if geom.pix_type == PixelShape.HEXAGON:
            xs, ys = generate_hex_vertices(geom)
        else:
            raise NotImplementedError(f"Unsupported pixel shape {geom.pix_type}")

        if image is None:
            image = np.zeros(geom.n_pixels)

        self.datasource = bokeh.plotting.ColumnDataSource(
            data=dict(
                poly_xs=xs.tolist(), poly_ys=ys.tolist(), image=image, id=geom.pix_id
            )
        )

        self._color_mapper = LinearColorMapper(
            palette=bokeh.palettes.Viridis256, low=image.min(), high=image.max()
        )

        frame = geom.frame.__class__.__name__ if geom.frame else "CameraFrame"
        self.figure = figure(
            title=f"{geom} ({frame})", match_aspect=True, aspect_scale=1
        )

        # Make sure the box zoom tool does not distort the camera display
        for tool in self.figure.toolbar.tools:
            if isinstance(tool, BoxZoomTool):
                tool.match_aspect = True

        self._setup_camera()

        if use_notebook is None:
            use_notebook = is_notebook()

        if use_notebook:
            output_notebook()

        self._use_notebook = use_notebook

        if autoshow:
            self._handle = show(self.figure, notebook_handle=use_notebook)

    def _setup_camera(self):
        self._pixels = self.figure.patches(
            xs="poly_xs",
            ys="poly_ys",
            fill_color=dict(field="image", transform=self._color_mapper),
            line_width=0,
            source=self.datasource,
        )

        self.figure.add_tools(HoverTool(tooltips=[("id", "@id"), ("value", "@image")]))

        self._color_bar = ColorBar(
            color_mapper=self._color_mapper,
            label_standoff=12,
            border_line_color=None,
            location=(0, 0),
        )
        self.figure.add_layout(self._color_bar, "right")

    def update(self):
        if self._use_notebook and self._handle:
            push_notebook(self._handle)

    def rescale(self, percent=100):
        self._color_mapper.update(
            low=self.datasource.data["image"].min(),
            high=(percent / 100) * self.datasource.data["image"].max(),
        )

    @property
    def cmap(self):
        return self._color_mapper.palette

    @cmap.setter
    def cmap(self, cmap):
        if isinstance(cmap, str):
            cmap = CMAPS[cmap]

        self._color_mapper.palette = cmap
        self.rescale()

    @property
    def image(self,):
        return self.datasource.data["image"]

    @image.setter
    def image(self, new_image):
        self.datasource.patch({"image": [(slice(None), new_image)]})
        self.rescale()
        self.update()


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
