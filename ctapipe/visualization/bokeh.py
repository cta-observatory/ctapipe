import warnings
import numpy as np
from bokeh.plotting import figure
from bokeh.events import Tap
from bokeh import palettes
from bokeh.models import (
    ColumnDataSource,
    TapTool,
    Span,
    ColorBar,
    LinearColorMapper,
)
from ctapipe.utils.rgbtohex import intensity_to_hex

PLOTARGS = dict(tools="", toolbar_location=None, outline_line_color='#595959')


class CameraDisplay:
    def __init__(self, geometry=None, image=None, fig=None):
        """
        Camera display that utilises the bokeh visualisation library

        Parameters
        ----------
        geometry : `~ctapipe.instrument.CameraGeometry`
            Definition of the Camera/Image
        image : ndarray
            1D array containing the image values for each pixel
        fig : bokeh.plotting.figure
            Figure to store the bokeh plot onto (optional)
        """
        self._geom = None
        self._image = None
        self._colors = None
        self._image_min = None
        self._image_max = None
        self._fig = None

        self._n_pixels = None
        self._pix_sizes = np.ones(1)
        self._pix_areas = np.ones(1)
        self._pix_x = np.zeros(1)
        self._pix_y = np.zeros(1)

        self.glyphs = None
        self.cm = None
        self.cb = None

        cdsource_d = dict(image=[],
                          x=[], y=[],
                          width=[], height=[],
                          outline_color=[], outline_alpha=[])
        self.cdsource = ColumnDataSource(data=cdsource_d)

        self._active_pixels = []
        self.active_index = 0
        self.active_colors = []
        self.automatic_index_increment = False

        self.geom = geometry
        self.image = image
        self.fig = fig

        self.layout = self.fig

    @property
    def fig(self):
        return self._fig

    @fig.setter
    def fig(self, val):
        if val is None:
            val = figure(plot_width=550, plot_height=500, **PLOTARGS)
        val.axis.visible = False
        val.grid.grid_line_color = None
        self._fig = val

        self._draw_camera()

    @property
    def geom(self):
        return self._geom

    @geom.setter
    def geom(self, val):
        self._geom = val

        if val is not None:
            self._pix_areas = val.pix_area.value
            self._pix_sizes = np.sqrt(self._pix_areas)
            self._pix_x = val.pix_x.value
            self._pix_y = val.pix_y.value

        self._n_pixels = self._pix_x.size
        if self._n_pixels == len(self.cdsource.data['x']):
            self.cdsource.data['x'] = self._pix_x
            self.cdsource.data['y'] = self._pix_y
            self.cdsource.data['width'] = self._pix_sizes
            self.cdsource.data['height'] = self._pix_sizes
        else:
            self._image = np.empty(self._pix_x.shape)
            alpha = [0] * self._n_pixels
            color = ['black'] * self._n_pixels
            cdsource_d = dict(image=self.image,
                              x=self._pix_x, y=self._pix_y,
                              width=self._pix_sizes, height=self._pix_sizes,
                              outline_color=color, outline_alpha=alpha
                              )
            self.cdsource.data = cdsource_d

        self.active_pixels = [0] * len(self.active_pixels)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, val):
        if val is None:
            val = np.zeros(self._n_pixels)

        image_min = np.nanmin(val)
        image_max = np.nanmax(val)
        if image_max == image_min:
            image_min -= 1
            image_max += 1
        colors = intensity_to_hex(val, image_min, image_max)

        self._image = val
        self._colors = colors
        self.image_min = image_min
        self.image_max = image_max

        if len(colors) == self._n_pixels:
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                self.cdsource.data['image'] = colors
        else:
            raise ValueError("Image has a different size {} than the current "
                             "CameraGeometry n_pixels {}"
                             .format(colors.size, self._n_pixels))

    @property
    def image_min(self):
        return self._image_min

    @image_min.setter
    def image_min(self, val):
        self._image_min = val
        if self.cb:
            self.cm.low = val.item()

    @property
    def image_max(self):
        return self._image_max

    @image_max.setter
    def image_max(self, val):
        self._image_max = val
        if self.cb:
            self.cm.high = val.item()

    @property
    def active_pixels(self):
        return self._active_pixels

    @active_pixels.setter
    def active_pixels(self, listval):
        self._active_pixels = listval

        palette = palettes.Set1[9]
        palette = [palette[0]] + palette[3:]
        self.active_colors = [palette[i % (len(palette))]
                              for i in range(len(listval))]
        self.highlight_pixels()

    def reset_pixels(self):
        self.active_pixels = [0] * len(self.active_pixels)

    def _draw_camera(self):
        # TODO: Support other pixel shapes OR switch to ellipse
        # after https://github.com/bokeh/bokeh/issues/6985
        self.glyphs = self.fig.ellipse(
            'x', 'y', color='image', width='width', height='height',
            line_color='outline_color',
            line_alpha='outline_alpha',
            line_width=2,
            nonselection_fill_color='image',
            nonselection_fill_alpha=1,
            nonselection_line_color='outline_color',
            nonselection_line_alpha='outline_alpha',
            source=self.cdsource
        )

    def enable_pixel_picker(self, n_active):
        """
        Enables the selection of a pixel by clicking on it

        Parameters
        ----------
        n_active : int
            Number of active pixels to keep record of
        """
        self.active_pixels = [0] * n_active
        self.fig.add_tools(TapTool())

        def source_change_response(_, __, val):
            if val:
                pix = val[0]
                ai = self.active_index
                self.active_pixels[ai] = pix

                self.highlight_pixels()
                self._on_pixel_click(pix)

                if self.automatic_index_increment:
                    self.active_index = (ai + 1) % len(self.active_pixels)

        self.cdsource.selected.on_change('indices', source_change_response)

    def _on_pixel_click(self, pix_id):
        print(f"Clicked pixel_id: {pix_id}")
        print(f"Active Pixels: {self.active_pixels}")

    def highlight_pixels(self):
        alpha = [0] * self._n_pixels
        color = ['black'] * self._n_pixels
        for i, pix in enumerate(self.active_pixels):
            alpha[pix] = 1
            color[pix] = self.active_colors[i]
        self.cdsource.data['outline_alpha'] = alpha
        self.cdsource.data['outline_color'] = color

    def add_colorbar(self):
        self.cm = LinearColorMapper(palette="Viridis256", low=0, high=100,
                                    low_color='white', high_color='red')
        self.cb = ColorBar(color_mapper=self.cm,
                           border_line_color=None,
                           background_fill_alpha=0,
                           major_label_text_color='green',
                           location=(0, 0))
        self.fig.add_layout(self.cb, 'right')
        self.cm.low = self.image_min.item()
        self.cm.high = self.image_max.item()


class FastCameraDisplay:
    def __init__(self, x_pix, y_pix, pix_size):
        """
        A fast and simple version of the bokeh camera plotter that does not
        allow for geometry changes

        Parameters
        ----------
        x_pix : ndarray
            Pixel x positions
        y_pix : ndarray
            Pixel y positions
        pix_size : ndarray
            Pixel sizes
        """
        self._image = None
        n_pix = x_pix.size

        cdsource_d = dict(image=np.empty(n_pix, dtype='<U8'), x=x_pix, y=y_pix)
        self.cdsource = ColumnDataSource(cdsource_d)
        self.fig = figure(plot_width=400, plot_height=400, **PLOTARGS)
        self.fig.grid.grid_line_color = None
        self.fig.rect('x', 'y', color='image', source=self.cdsource,
                      width=pix_size[0], height=pix_size[0])

        self.layout = self.fig

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, val):
        """
        Parameters
        ----------
        val : ndarray
            Array containing the image values, already converted into
            hexidecimal strings
        """
        self.cdsource.data['image'] = val


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

        if len(val) == len(self.cdsource.data['t']):
            self.cdsource.data['samples'] = val
        else:
            cdsource_d = dict(t=np.arange(val.size), samples=val)
            self.cdsource.data = cdsource_d

    @property
    def active_time(self):
        return self._active_time

    @active_time.setter
    def active_time(self, val):
        max_t = self.cdsource.data['t'][-1]
        if val is None:
            val = 0
        if val < 0:
            val = 0
        if val > max_t:
            val = max_t
        self.span.location = val
        self._active_time = val

    def _draw_waveform(self):
        self.fig.line(x="t", y="samples", source=self.cdsource, name='line')

    def enable_time_picker(self):
        """
        Enables the selection of a time by clicking on the waveform
        """
        self.span = Span(location=0, dimension='height',
                         line_color='red', line_dash='dashed')
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
