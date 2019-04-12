"""
Calibrate dl0 data to dl1, and plot the photoelectron images.
"""
from matplotlib import pyplot as plt, colors
from matplotlib.backends.backend_pdf import PdfPages
from traitlets import Dict, List, Int, Bool, Unicode

from ctapipe.calib import CameraCalibrator, CameraDL1Calibrator
from ctapipe.visualization import CameraDisplay
from ctapipe.core import Tool, Component
from ctapipe.utils import get_dataset_path
from ctapipe.image.extractor import ImageExtractor
from ctapipe.io import EventSource
import ctapipe.utils.tools as tool_utils


class ImagePlotter(Component):
    display = Bool(
        True,
        help='Display the photoelectron images on-screen as they '
             'are produced.'
    ).tag(config=True)
    output_path = Unicode(
        None,
        allow_none=True,
        help='Output path for the pdf containing all the '
             'images. Set to None for no saved '
             'output.'
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
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
        self.c_pulse_time = None
        self.cb_intensity = None
        self.cb_pulse_time = None
        self.pdf = None

        self._init_figure()

    def _init_figure(self):
        self.fig = plt.figure(figsize=(16, 7))
        self.ax_intensity = self.fig.add_subplot(1, 2, 1)
        self.ax_pulse_time = self.fig.add_subplot(1, 2, 2)
        if self.output_path:
            self.log.info(f"Creating PDF: {self.output_path}")
            self.pdf = PdfPages(self.output_path)

    @staticmethod
    def get_geometry(event, telid):
        return event.inst.subarray.tel[telid].camera

    def plot(self, event, telid):
        chan = 0
        image = event.dl1.tel[telid].image[chan]
        pulse_time = event.dl1.tel[telid].pulse_time[chan]

        if self._current_tel != telid:
            self._current_tel = telid

            self.ax_intensity.cla()
            self.ax_pulse_time.cla()

            # Redraw camera
            geom = self.get_geometry(event, telid)
            self.c_intensity = CameraDisplay(geom, ax=self.ax_intensity)
            self.c_pulse_time = CameraDisplay(geom, ax=self.ax_pulse_time)

            tmaxmin = event.dl0.tel[telid].waveform.shape[2]
            t_chargemax = pulse_time[image.argmax()]
            cmap_time = colors.LinearSegmentedColormap.from_list(
                'cmap_t',
                [(0 / tmaxmin, 'darkgreen'),
                 (0.6 * t_chargemax / tmaxmin, 'green'),
                 (t_chargemax / tmaxmin, 'yellow'),
                 (1.4 * t_chargemax / tmaxmin, 'blue'), (1, 'darkblue')]
            )
            self.c_pulse_time.pixels.set_cmap(cmap_time)

            if not self.cb_intensity:
                self.c_intensity.add_colorbar(
                    ax=self.ax_intensity, label='Intensity (p.e.)'
                )
                self.cb_intensity = self.c_intensity.colorbar
            else:
                self.c_intensity.colorbar = self.cb_intensity
                self.c_intensity.update(True)
            if not self.cb_pulse_time:
                self.c_pulse_time.add_colorbar(
                    ax=self.ax_pulse_time, label='Pulse Time (ns)'
                )
                self.cb_pulse_time = self.c_pulse_time.colorbar
            else:
                self.c_pulse_time.colorbar = self.cb_pulse_time
                self.c_pulse_time.update(True)

        self.c_intensity.image = image
        if pulse_time is not None:
            self.c_pulse_time.image = pulse_time

        self.fig.suptitle(
            "Event_index={}  Event_id={}  Telescope={}"
                .format(event.count, event.r0.event_id, telid)
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
        help='Telescope to view. Set to None to display all '
             'telescopes.'
    ).tag(config=True)

    extractor_product = tool_utils.enum_trait(
        ImageExtractor,
        default='NeighborPeakWindowSum'
    )

    aliases = Dict(
        dict(
            max_events='EventSource.max_events',
            extractor='DisplayDL1Calib.extractor_product',
            T='DisplayDL1Calib.telescope',
            O='ImagePlotter.output_path'
        )
    )
    flags = Dict(
        dict(
            D=(
                {
                    'ImagePlotter': {
                        'display': True
                    }
                },
                "Display the photoelectron images on-screen as they "
                "are produced."
            )
        )
    )
    classes = List(
        [
            EventSource,
            CameraDL1Calibrator,
            ImagePlotter
        ] + tool_utils.classes_with_traits(ImageExtractor)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eventsource = None
        self.calibrator = None
        self.plotter = None

    def setup(self):
        self.eventsource = EventSource.from_url(
            get_dataset_path("gamma_test_large.simtel.gz"),
            parent=self,
        )

        self.calibrator = CameraCalibrator(parent=self)

        self.plotter = ImagePlotter(parent=self)

    def start(self):
        for event in self.eventsource:
            self.calibrator.calibrate(event)

            tel_list = event.r0.tels_with_data

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
