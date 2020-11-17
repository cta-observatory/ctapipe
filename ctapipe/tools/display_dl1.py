"""
Calibrate dl0 data to dl1, and plot the photoelectron images.
"""
from copy import copy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from traitlets import Bool, Dict, Int, List

from ctapipe.calib import CameraCalibrator
from ctapipe.core import Component, Tool
from ctapipe.core import traits
from ctapipe.image.extractor import ImageExtractor
from ctapipe.io import EventSource
from ctapipe.utils import get_dataset_path
from ctapipe.visualization import CameraDisplay


class ImagePlotter(Component):
    """ Plotter for camera images """

    display = Bool(
        True, help="Display the photoelectron images on-screen as they are produced."
    ).tag(config=True)
    output_path = traits.Path(
        directory_ok=False,
        help=(
            "Output path for the pdf containing all the images."
            " Set to None for no saved output."
        ),
    ).tag(config=True)

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        """
        Plotter for camera images.

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, parent=parent, **kwargs)
        self._current_tel = None
        self.c_intensity = None
        self.c_peak_time = None
        self.cb_intensity = None
        self.cb_peak_time = None
        self.pdf = None
        self.subarray = subarray

        self._init_figure()

    def _init_figure(self):
        self.fig = plt.figure(figsize=(16, 7))
        self.ax_intensity = self.fig.add_subplot(1, 2, 1)
        self.ax_peak_time = self.fig.add_subplot(1, 2, 2)
        if self.output_path:
            self.log.info(f"Creating PDF: {self.output_path}")
            self.pdf = PdfPages(self.output_path)

    def plot(self, event, telid):
        image = event.dl1.tel[telid].image
        peak_time = event.dl1.tel[telid].peak_time

        if self._current_tel != telid:
            self._current_tel = telid

            self.ax_intensity.cla()
            self.ax_peak_time.cla()

            # Redraw camera
            geom = self.subarray.tel[telid].camera.geometry
            self.c_intensity = CameraDisplay(geom, ax=self.ax_intensity)

            time_cmap = copy(plt.get_cmap("RdBu_r"))
            time_cmap.set_under("gray")
            time_cmap.set_over("gray")
            self.c_peak_time = CameraDisplay(geom, ax=self.ax_peak_time, cmap=time_cmap)

            if not self.cb_intensity:
                self.c_intensity.add_colorbar(
                    ax=self.ax_intensity, label="Intensity (p.e.)"
                )
                self.cb_intensity = self.c_intensity.colorbar
            else:
                self.c_intensity.colorbar = self.cb_intensity
                self.c_intensity.update(True)
            if not self.cb_peak_time:
                self.c_peak_time.add_colorbar(
                    ax=self.ax_peak_time, label="Pulse Time (ns)"
                )
                self.cb_peak_time = self.c_peak_time.colorbar
            else:
                self.c_peak_time.colorbar = self.cb_peak_time
                self.c_peak_time.update(True)

        self.c_intensity.image = image
        self.c_peak_time.image = peak_time

        # center around brightes pixel, show 10ns total
        t_chargemax = peak_time[image.argmax()]
        self.c_peak_time.set_limits_minmax(t_chargemax - 5, t_chargemax + 5)

        self.fig.suptitle(
            "Event_index={}  Event_id={}  Telescope={}".format(
                event.count, event.index.event_id, telid
            )
        )

        if self.display:
            plt.pause(0.001)

        if self.pdf is not None:
            self.pdf.savefig(self.fig)

    def finish(self):
        if self.pdf is not None:
            self.log.info("Closing PDF")
            self.pdf.close()


class DisplayDL1Calib(Tool):
    name = "ctapipe-display-dl1"
    description = __doc__

    telescope = Int(
        None,
        allow_none=True,
        help="Telescope to view. Set to None to display all telescopes.",
    ).tag(config=True)

    aliases = Dict(
        dict(
            input="EventSource.input_url",
            max_events="EventSource.max_events",
            T="DisplayDL1Calib.telescope",
            O="ImagePlotter.output_path",
        )
    )
    flags = Dict(
        dict(
            D=(
                {"ImagePlotter": {"display": True}},
                "Display the photo-electron images on-screen as they are produced.",
            )
        )
    )
    classes = List(
        [EventSource, ImagePlotter] + traits.classes_with_traits(ImageExtractor)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.EventSource.input_url = get_dataset_path(
            "gamma_test_large.simtel.gz"
        )
        self.eventsource = None
        self.calibrator = None
        self.plotter = None

    def setup(self):
        self.eventsource = EventSource(parent=self)
        subarray = self.eventsource.subarray

        self.calibrator = CameraCalibrator(parent=self, subarray=subarray)
        self.plotter = ImagePlotter(parent=self, subarray=subarray)

    def start(self):
        for event in self.eventsource:
            self.calibrator(event)

            tel_list = event.r0.tel.keys()

            if self.telescope:
                if self.telescope not in tel_list:
                    continue
                tel_list = [self.telescope]

            for telid in tel_list:
                self.plotter.plot(event, telid)

    def finish(self):
        self.plotter.finish()


def main():
    exe = DisplayDL1Calib()
    exe.run()
