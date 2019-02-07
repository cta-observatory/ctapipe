"""Module to handle plotting of camera event specific items, e.g.
camera images and waveforms.
"""

from matplotlib import pyplot as plt

from ctapipe.visualization import CameraDisplay

from astropy import units as u
from astropy import log


class CameraPlotter:
    """
    Plotter object for basic items that are camera event specific.

    WARNING: a new CameraPlotter object must be created for each event

    Attributes
    ----------
    event : container
        A `ctapipe` event container
    """

    def __init__(self, event):
        """
        Parameters
        ----------
        event : container
            A `ctapipe` event container
        """
        self.event = event
        self.cameradisplay_dict = {}

    def get_geometry(self, tel):
        return self.event.inst.subarray.tel[tel].camera

    def draw_camera(self, tel, data, axes=None):
        """
        Draw a camera image using the correct geometry.

        Parameters
        ----------
        tel : int
            The telescope you want drawn.
        data : `np.array`
            1D array with length equal to npix.
        axes : `matplotlib.axes.Axes`
            A matplotlib axes object to plot on, or None to create a new one.

        Returns
        -------
        `ctapipe.visualization.CameraDisplay`
        """

        geom = self.get_geometry(tel)
        axes = axes if axes is not None else plt.gca()
        camera = CameraDisplay(geom, ax=axes)
        camera.image = data
        camera.cmap = plt.cm.viridis
        # camera.add_colorbar(ax=axes, label="Amplitude (ADC)")
        # camera.set_limits_percent(95)  # autoscale
        return camera

    def draw_camera_pixel_ids(self, tel, pixels, axes=None):
        """
        Draw the pixel_ids on top of a camera image

        Parameters
        ----------
        tel : int
            The telescope you want drawn.
        pixels : list
            A list of the pixel IDs you want drawing
        axes : `matplotlib.axes.Axes`
            A matplotlib axes object to plot on, or None to create a new one.
        """

        geom = self.get_geometry(tel)
        axes = axes if axes is not None else plt.gca()
        log.info("Annotating with pixel_ids")
        for pix in pixels:
            x = u.Quantity(geom.pix_x).value[pix]
            y = u.Quantity(geom.pix_y).value[pix]
            axes.text(x, y, pix, fontsize=2, ha='center')

    def draw_camera_pixel_annotation(self, tel, p0, p1, axes=None):
        """
        Draw annotations for two pixels, used for camera images that are
        accompanied by two waveforms

        Parameters
        ----------
        tel : int
            The telescope you want drawn.
        p0 : int
            First pixel_id
        p1 : int
            Second pixel_id
        axes : `matplotlib.axes.Axes`
            A matplotlib axes object to plot on, or None to create a new one.
        """

        geom = self.get_geometry(tel)
        axes = axes if axes is not None else plt.gca()
        axes.annotate(f"Pixel: {p0}",
                      xy=(u.Quantity(geom.pix_x).value[p0],
                          u.Quantity(geom.pix_y).value[p0]),
                      xycoords='data', xytext=(0.05, 0.98),
                      textcoords='axes fraction',
                      arrowprops=dict(facecolor='red', width=2, alpha=0.4),
                      horizontalalignment='left', verticalalignment='top')
        axes.annotate(f"Pixel: {p1}",
                      xy=(u.Quantity(geom.pix_x).value[p1],
                          u.Quantity(geom.pix_y).value[p1]),
                      xycoords='data', xytext=(0.05, 0.94),
                      textcoords='axes fraction',
                      arrowprops=dict(facecolor='orange', width=2, alpha=0.4),
                      horizontalalignment='left', verticalalignment='top')

    @staticmethod
    def draw_waveform(data, axes=None):
        """
        Draw the waveform from a pixel.

        Parameters
        ----------
        data : `np.array`
            1D array with length equal to a waveforms sample size.
        axes : `matplotlib.axes.Axes`
            A matplotlib axes object to plot on, or None to create a new one.

        Returns
        -------
        `matplotlib.lines.Line2D`

        """

        axes = axes if axes is not None else plt.gca()
        waveform, = axes.plot(data)
        axes.set_xlabel("Time (ns)")
        axes.set_ylabel("Amplitude (ADC)")
        return waveform

    @staticmethod
    def draw_waveform_positionline(t, axes=None):
        """
        Draw a vertical line on a waveform. Intended to be used for showing
        postion along a waveform for a animation or the event.
        Parameters
        ----------
        t : float
            X position (time) for the line to be drawn at on the waveform.
        axes : `matplotlib.axes.Axes`
            A matplotlib axes object to plot on, or None to create a new one.

        Returns
        -------
        `matplotlib.lines.Line2D`

        """
        axes = axes if axes is not None else plt.gca()
        line, = axes.plot([t, t], axes.get_ylim(), color='r', alpha=1)
        return line
