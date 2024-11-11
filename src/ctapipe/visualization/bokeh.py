import sys
from abc import ABCMeta
from tempfile import NamedTemporaryFile

import astropy.units as u
import numpy as np

from ..exceptions import OptionalDependencyMissing
from ..instrument import CameraGeometry, PixelShape
from .utils import build_hillas_overlay

try:
    from bokeh.io import output_file, output_notebook, push_notebook, show
    from bokeh.models import (
        BoxZoomTool,
        CategoricalColorMapper,
        ColorBar,
        ColumnDataSource,
        ContinuousColorMapper,
        Ellipse,
        HoverTool,
        Label,
        LinearColorMapper,
        LogColorMapper,
        TapTool,
    )
    from bokeh.palettes import Greys256, Inferno256, Magma256, Viridis256, d3
    from bokeh.plotting import figure
    from bokeh.transform import transform
except ModuleNotFoundError:
    raise OptionalDependencyMissing("bokeh") from None


__all__ = [
    "BokehPlot",
    "CameraDisplay",
    "ArrayDisplay",
]


PLOTARGS = dict(tools="", toolbar_location=None, outline_line_color="#595959")

# mapper to mpl names
CMAPS = {
    "viridis": Viridis256,
    "magma": Magma256,
    "inferno": Inferno256,
    "grey": Greys256,
    "gray": Greys256,
}


def palette_from_mpl_name(name):
    """Create a bokeh palette from a matplotlib colormap name"""
    if name in CMAPS:
        return CMAPS[name]

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex
    except ModuleNotFoundError:
        raise OptionalDependencyMissing("matplotlib") from None

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
    """Generate vertices of pixels for a hexagonal grid camera geometry"""
    phi = np.arange(0, 2 * np.pi, np.pi / 3)

    # apply pixel rotation and conversion from flat top to pointy top
    phi += geom.pix_rotation.rad + np.deg2rad(30)

    # we need the circumcircle radius, pixel_width is incircle diameter
    unit = geom.pix_x.unit
    r = 2 / np.sqrt(3) * geom.pixel_width.to_value(unit) / 2

    x = geom.pix_x.to_value(unit)
    y = geom.pix_y.to_value(unit)

    return (
        x[:, np.newaxis] + r[:, np.newaxis] * np.cos(phi)[np.newaxis],
        y[:, np.newaxis] + r[:, np.newaxis] * np.sin(phi)[np.newaxis],
    )


def generate_square_vertices(geom):
    """Generate vertices of pixels for a square grid camera geometry"""
    unit = geom.pix_x.unit
    width = geom.pixel_width.to_value(unit) / 2
    x = geom.pix_x.to_value(unit)
    y = geom.pix_y.to_value(unit)

    x_offset = width[:, np.newaxis] * np.array([-1, -1, 1, 1])
    y_offset = width[:, np.newaxis] * np.array([1, -1, -1, 1])

    x = x[:, np.newaxis] + x_offset
    y = y[:, np.newaxis] + y_offset
    return x, y


class BokehPlot(metaclass=ABCMeta):
    """Base class for bokeh plots"""

    def __init__(
        self,
        use_notebook=None,
        autoscale=True,
        cmap="inferno",
        norm="lin",
        **figure_kwargs,
    ):
        # only use autoshow / use_notebook by default if we are in a notebook
        self._use_notebook = use_notebook if use_notebook is not None else is_notebook()
        self._patches = None
        self._handle = None
        self._color_bar = None
        self._color_mapper = None
        self._palette = None
        self._annotations = []
        self._labels = []

        self.cmap = cmap
        self.norm = norm

        self.autoscale = autoscale

        self.datasource = ColumnDataSource(data={})
        self.figure = figure(**figure_kwargs)
        self.figure.add_tools(HoverTool(tooltips=[("id", "@id"), ("value", "@values")]))

        if figure_kwargs.get("match_aspect"):
            # Make sure the box zoom tool does not distort the camera display
            for tool in self.figure.toolbar.tools:
                if isinstance(tool, BoxZoomTool):
                    tool.match_aspect = True

    def show(self):
        """Display the figure"""
        if self._use_notebook:
            output_notebook()
        else:
            # this only sets the default name, created only when show is called
            tmp = NamedTemporaryFile(
                delete=False, prefix="ctapipe_bokeh_", suffix=".html"
            )
            output_file(tmp.name)

        self._handle = show(self.figure, notebook_handle=self._use_notebook)

    def update(self):
        """Update the figure"""
        if self._use_notebook and self._handle:
            push_notebook(handle=self._handle)

    def add_colorbar(self):
        self._color_bar = ColorBar(
            color_mapper=self._color_mapper,
            label_standoff=12,
            border_line_color=None,
            location=(0, 0),
        )
        self.figure.add_layout(self._color_bar, "right")
        self.update()

    def set_limits_minmax(self, zmin, zmax):
        """Set the limits of the color range to ``zmin`` / ``zmax``"""
        self._color_mapper.update(low=zmin, high=zmax)
        self.update()

    def set_limits_percent(self, percent=95):
        """Set the limits to min / fraction of max value"""
        low = np.nanmin(self.datasource.data["values"])
        high = np.nanmax(self.datasource.data["values"])

        frac = percent / 100.0
        self.set_limits_minmax(low, high - (1.0 - frac) * (high - low))

    @property
    def cmap(self):
        """Get the current colormap"""
        return self._palette

    @cmap.setter
    def cmap(self, cmap):
        """Set colormap"""
        if isinstance(cmap, str):
            cmap = palette_from_mpl_name(cmap)

        self._palette = cmap
        # might be called in __init__ before color mapper is setup
        if self._color_mapper is not None:
            self._color_mapper.palette = cmap
            self._trigger_cm_update()
            self.update()

    def _trigger_cm_update(self):
        # it seems changing palette does not trigger a color change,
        # so we reassign a property
        if isinstance(self._color_mapper, CategoricalColorMapper):
            self._color_mapper.update(factors=self._color_mapper.factors)
        else:
            self._color_mapper.update(low=self._color_mapper.low)

    def clear_overlays(self):
        """Remove any added overlays from the figure"""
        while self._annotations:
            self.figure.renderers.remove(self._annotations.pop())

        while self._labels:
            self.figure.center.remove(self._labels.pop())

    def rescale(self):
        """Scale pixel colors to min/max range"""
        low = self.datasource.data["values"].min()
        high = self.datasource.data["values"].max()

        # force color to be at lower end of the colormap if
        # data is all equal
        if low == high:
            high += 1

        self.set_limits_minmax(low, high)

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
    def norm(self, norm):
        """Set the norm"""
        if not isinstance(norm, ContinuousColorMapper):
            if norm == "lin":
                norm = LinearColorMapper
            elif norm == "log":
                norm = LogColorMapper
            else:
                raise ValueError(f"Unsupported norm {norm}")

        self._color_mapper = norm(palette=self.cmap)
        if self._patches is not None:
            color = transform("values", self._color_mapper)
            self._patches.glyph.update(fill_color=color)

        if self._color_bar is not None:
            self._color_bar.update(color_mapper=self._color_mapper)

        self.update()


class CameraDisplay(BokehPlot):
    """
    CameraDisplay implementation in Bokeh

    Parameters
    ----------
    geometry: CameraGeometry
        CameraGeometry for the display
    image: array
        Values to display for each pixel
    cmap: str or bokeh.palette.Palette
        matplotlib colormap name or bokeh palette for color mapping values
    norm: str or bokeh.
        lin, log, symlog or a bokeh.models.ColorMapper instance
    autoscale: bool
        Whether to automatically adjust color range after updating image
    use_notebook: bool or None
        Whether to use bokehs notebook output. If None, tries to autodetect
        running in a notebook.

    **figure_kwargs are passed to bokeh.plots.figure
    """

    def __init__(
        # same options as MPL display
        self,
        geometry: CameraGeometry = None,
        image=None,
        cmap="inferno",
        norm="lin",
        autoscale=True,
        title=None,
        # bokeh specific options
        use_notebook=None,
        **figure_kwargs,
    ):
        super().__init__(
            use_notebook=use_notebook,
            cmap=cmap,
            norm=norm,
            autoscale=autoscale,
            title=title,
            match_aspect=True,
            aspect_scale=1,
            **figure_kwargs,
        )

        self._geometry = geometry
        self._tap_tool = None

        if geometry is not None:
            self._init_datasource(image)

            if title is None:
                frame = (
                    geometry.frame.__class__.__name__
                    if geometry.frame
                    else "CameraFrame"
                )
                title = f"{geometry} ({frame})"
            self.figure.title = title

        # order is important because steps depend on each other
        self.cmap = cmap
        self.norm = norm
        if geometry is not None:
            self.rescale()
            self._setup_camera()

    def _init_datasource(self, image=None):
        if image is None:
            image = np.zeros(self._geometry.n_pixels)

        data = dict(
            id=self._geometry.pix_id,
            values=image,
            line_width=np.zeros(self._geometry.n_pixels),
            line_color=["green"] * self._geometry.n_pixels,
            line_alpha=np.zeros(self._geometry.n_pixels),
        )

        self._unit = self._geometry.pix_x.unit

        if self._geometry.pix_type == PixelShape.HEXAGON:
            x, y = generate_hex_vertices(self._geometry)

        elif self._geometry.pix_type == PixelShape.SQUARE:
            x, y = generate_square_vertices(self._geometry)

        elif self._geometry.pix_type == PixelShape.CIRCLE:
            x = self._geometry.pix_x.to_value(self._unit)
            y = self._geometry.pix_y.to_value(self._unit)
            data["radius"] = self._geometry.pixel_width.to_value(self._unit) / 2
        else:
            raise NotImplementedError(
                f"Unsupported pixel shape {self._geometry.pix_type}"
            )

        data["xs"], data["ys"] = x.tolist(), y.tolist()

        self.datasource.update(data=data)

    def _setup_camera(self):
        kwargs = dict(
            fill_color=dict(field="values", transform=self.norm),
            line_width="line_width",
            line_color="line_color",
            line_alpha="line_alpha",
            source=self.datasource,
        )
        if self._geometry.pix_type in (PixelShape.SQUARE, PixelShape.HEXAGON):
            self._patches = self.figure.patches(xs="xs", ys="ys", **kwargs)
        elif self._geometry.pix_type == PixelShape.CIRCLE:
            self._patches = self.figure.circle(
                x="xs", y="ys", radius="radius", **kwargs
            )

    def enable_pixel_picker(self, callback):
        """Call ``callback`` when a pixel is clicked"""
        if self._tap_tool is None:
            self.figure.add_tools(TapTool())
        self.datasource.selected.on_change("indices", callback)

    def highlight_pixels(self, pixels, color="green", linewidth=1, alpha=0.75):
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
        n_pixels = self._geometry.n_pixels
        pixels = np.asanyarray(pixels)

        if pixels.dtype != bool:
            selected = np.zeros(n_pixels, dtype=bool)
            selected[pixels] = True
            pixels = selected

        new_data = {"line_alpha": [(slice(None), pixels.astype(float) * alpha)]}
        if linewidth != self.datasource.data["line_width"][0]:
            new_data["line_width"] = [(slice(None), np.full(n_pixels, linewidth))]

        if color != self.datasource.data["line_color"][0]:
            new_data["line_color"] = [(slice(None), [color] * n_pixels)]

        self.datasource.patch(new_data)
        self.update()

    @property
    def geometry(self):
        """Get the current geometry"""
        return self._geometry

    @geometry.setter
    def geometry(self, new_geometry):
        """Set the geometry"""
        self._geometry = new_geometry
        if self._patches in self.figure.renderers:
            self.figure.renderers.remove(self._patches)
        self._init_datasource()
        self._setup_camera()
        self.rescale()
        self.update()

    @property
    def image(self):
        """Get the current image"""
        return self.datasource.data["values"]

    @image.setter
    def image(self, new_image):
        """Set the image"""
        self.datasource.patch({"values": [(slice(None), new_image)]})
        if self.autoscale:
            self.rescale()

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
        kwargs: key=value
            any style keywords to pass to matplotlib (e.g. color='red'
            or linewidth=6)
        """
        if not keep_old:
            self.clear_overlays()

        params = build_hillas_overlay(
            hillas_parameters,
            self._unit,
            n_sigma=n_sigma,
            with_label=with_label,
        )

        el = self.add_ellipse(
            centroid=(params["cog_x"], params["cog_y"]),
            length=2 * n_sigma * params["length"],
            width=2 * n_sigma * params["width"],
            angle=params["psi_rad"],
            **kwargs,
        )

        if with_label:
            label = Label(
                x=params["label_x"],
                y=params["label_y"],
                text=params["text"],
                angle=params["rotation"],
                angle_units="deg",
                text_align="center",
                text_color=el.line_color,
            )
            self.figure.add_layout(label, "center")
            self._labels.append(label)


class ArrayDisplay(BokehPlot):
    """
    Display a top-town view of a telescope array.

    This can be used in two ways: by default, you get a display of all
    telescopes in the subarray, colored by telescope type, however you can
    also color the telescopes by a value (like trigger pattern, or some other
    scalar per-telescope parameter). To set the color value, simply set the
    ``value`` attribute, and the fill color will be updated with the value. You
    might want to set the border color to zero to avoid confusion between the
    telescope type color and the value color (
    ``array_disp.telescope.set_linewidth(0)``)

    To display a vector field over the telescope positions, e.g. for
    reconstruction, call ``set_uv()`` to set cartesian vectors, or ``set_r_phi()``
    to set polar coordinate vectors.  These both take an array of length
    N_tels, or a single value.


    Parameters
    ----------
    subarray: ctapipe.instrument.SubarrayDescription
        the array layout to display
    values: array
        Value to display for each telescope. If None, the telescope type will
        be used.
    scale: float
        scaling between telescope mirror radius in m to displayed size
    title: str
        title of array plot
    alpha: float
        Alpha value for telescopes
    cmap: str or bokeh.palette.Palette
        matplotlib colormap name or bokeh palette for color mapping values
    norm: str or bokeh.
        lin, log, symlog or a bokeh.models.ColorMapper instance
    radius: Union[float, list, None]
        set telescope radius to value, list/array of values. If None, radius
        is taken from the telescope's mirror size.
    use_notebook: bool or None
        Whether to use bokehs notebook output. If None, tries to autodetect
        running in a notebook.
    frame: GroundFrame or TiltedGroundFrame
        If given, transform telescope positions into this frame

    **figure_kwargs are passed to bokeh.plots.figure
    """

    def __init__(
        self,
        subarray,
        values=None,
        scale=5.0,
        alpha=1.0,
        title=None,
        cmap=None,
        norm="lin",
        radius=None,
        use_notebook=None,
        frame=None,
        **figure_kwargs,
    ):
        if title is None:
            frame_name = (frame or subarray.tel_coords.frame).__class__.__name__
            title = f"{subarray.name} ({frame_name})"

        # color by type if no value given
        if values is None:
            types = list({str(t) for t in subarray.telescope_types})
            cmap = cmap or d3["Category10"][10][: len(types)]
            field = "type"
        else:
            cmap = "inferno"
            field = "values"

        super().__init__(
            use_notebook=use_notebook,
            title=title,
            match_aspect=True,
            aspect_scale=1,
            cmap=cmap,
            norm=norm,
            **figure_kwargs,
        )

        if values is None:
            self._color_mapper = CategoricalColorMapper(palette=cmap, factors=types)

        self.frame = frame
        self.subarray = subarray

        self._init_datasource(
            subarray,
            values=values,
            radius=radius,
            frame=frame,
            scale=scale,
            alpha=alpha,
        )

        color = transform(field_name=field, transform=self._color_mapper)
        self._patches = self.figure.circle(
            x="x",
            y="y",
            radius="radius",
            alpha="alpha",
            line_alpha="alpha",
            fill_color=color,
            line_color=color,
            source=self.datasource,
            legend_field="type",
        )
        self.figure.add_tools(
            HoverTool(tooltips=[("id", "@id"), ("type", "@type"), ("z", "@z")])
        )
        self.figure.legend.orientation = "horizontal"
        self.figure.legend.location = "top_left"

    def _init_datasource(self, subarray, values, *, radius, frame, scale, alpha):
        telescope_ids = subarray.tel_ids
        tel_coords = subarray.tel_coords

        # get the telescope positions. If a new frame is set, this will
        # transform to the new frame.
        if frame is not None:
            tel_coords = tel_coords.transform_to(frame)

        tel_types = []
        mirror_radii = np.zeros(len(telescope_ids))

        for i, telescope_id in enumerate(telescope_ids):
            telescope = subarray.tel[telescope_id]
            tel_types.append(str(telescope))
            mirror_area = telescope.optics.mirror_area.to_value(u.m**2)
            mirror_radii[i] = np.sqrt(mirror_area) / np.pi

        if values is None:
            values = np.zeros(len(subarray))

        if np.isscalar(alpha):
            alpha = np.full(len(telescope_ids), alpha)
        else:
            alpha = np.array(alpha)

        data = {
            "id": telescope_ids,
            "x": tel_coords.x.to_value(u.m).tolist(),
            "y": tel_coords.y.to_value(u.m).tolist(),
            "z": tel_coords.z.to_value(u.m).tolist(),
            "alpha": alpha.tolist(),
            "values": values,
            "type": tel_types,
            "mirror_radius": mirror_radii.tolist(),
            "radius": (radius if radius is not None else mirror_radii * scale).tolist(),
        }

        self.datasource.update(data=data)

    @property
    def values(self):
        """Get the current image"""
        return self.datasource.data["values"]

    @values.setter
    def values(self, new_values):
        """Set the image"""
        # currently displaying telescope types
        if self._patches.glyph.fill_color["field"] == "type":
            self.norm = "lin"
            self.cmap = "inferno"
            color = transform(field_name="values", transform=self._color_mapper)
            self._patches.glyph.update(fill_color=color, line_color=color)

            # recreate color bar, updating does not work here
            if self._color_bar is not None:
                self.figure.right.remove(self._color_bar)
                self.add_colorbar()

            self.update()

        self.datasource.patch({"values": [(slice(None), new_values)]})
        if self.autoscale:
            self.rescale()
