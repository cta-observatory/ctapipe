from traitlets import Dict, List, Int, Bool, Unicode
from matplotlib import pyplot as plt, colors
from matplotlib.backends.backend_pdf import PdfPages
from ctapipe.core import Tool, Component
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera import CameraCalibrator
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay


class ImagePlotter(Component):
    name = 'ImagePlotter'

    display = Bool(False,
                   help='Display the photoelectron images on-screen as they '
                        'are produced.').tag(config=True)
    output_path = Unicode(None, allow_none=True,
                          help='Output path for the pdf containing all the '
                               'images. Set to None for no saved '
                               'output.').tag(config=True)

    def __init__(self, config, tool, **kwargs):
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
        super().__init__(config=config, parent=tool, **kwargs)
        self._geom_dict = {}
        self._current_tel = None
        self.c_intensity = None
        self.c_peakpos = None
        self.cb_intensity = None
        self.cb_peakpos = None
        self.pdf = None

        self._init_figure()

    def _init_figure(self):
        self.fig = plt.figure(figsize=(16, 7))
        self.ax_intensity = self.fig.add_subplot(1, 2, 1)
        self.ax_peakpos = self.fig.add_subplot(1, 2, 2)
        if self.output_path:
            self.log.info("Creating PDF: {}".format(self.output_path))
            self.pdf = PdfPages(self.output_path)

    def get_geometry(self, event, telid):
        if telid not in self._geom_dict:
            geom = CameraGeometry.guess(*event.inst.pixel_pos[telid],
                                        event.inst.optical_foclen[telid])
            self._geom_dict[telid] = geom
        return self._geom_dict[telid]

    def plot(self, event, telid):
        chan = 0
        image = event.dl1.tel[telid].image[chan]
        peakpos = event.dl1.tel[telid].peakpos[chan]

        if self._current_tel != telid:
            self._current_tel = telid

            self.ax_intensity.cla()
            self.ax_peakpos.cla()

            # Redraw camera
            geom = self.get_geometry(event, telid)
            self.c_intensity = CameraDisplay(geom, cmap=plt.cm.viridis,
                                             ax=self.ax_intensity)
            self.c_peakpos = CameraDisplay(geom, cmap=plt.cm.viridis,
                                           ax=self.ax_peakpos)

            tmaxmin = event.dl0.tel[telid].pe_samples.shape[2]
            t_chargemax = peakpos[image.argmax()]
            cmap_time = colors.LinearSegmentedColormap.from_list(
                'cmap_t', [(0 / tmaxmin, 'darkgreen'),
                           (0.6 * t_chargemax / tmaxmin, 'green'),
                           (t_chargemax / tmaxmin, 'yellow'),
                           (1.4 * t_chargemax / tmaxmin, 'blue'),
                           (1, 'darkblue')])
            self.c_peakpos.pixels.set_cmap(cmap_time)

            if not self.cb_intensity:
                self.c_intensity.add_colorbar(ax=self.ax_intensity,
                                              label='Intensity (p.e.)')
                self.cb_intensity = self.c_intensity.colorbar
            else:
                self.c_intensity.colorbar = self.cb_intensity
                self.c_intensity.update(True)
            if not self.cb_peakpos:
                self.c_peakpos.add_colorbar(ax=self.ax_peakpos,
                                            label='Peakpos (ns)')
                self.cb_peakpos = self.c_peakpos.colorbar
            else:
                self.c_peakpos.colorbar = self.cb_peakpos
                self.c_peakpos.update(True)

        self.c_intensity.image = image
        if peakpos is not None:
            self.c_peakpos.image = peakpos

        self.fig.suptitle("Event_index={}  Event_id={}  Telescope={}"
                          .format(event.count, event.r0.event_id, telid))


        if self.display:
            plt.pause(0.001)
        if self.pdf is not None:
            self.pdf.savefig(self.fig)

    def finish(self):
        if self.pdf is not None:
            self.log.info("Closing PDF")
            self.pdf.close()


class DisplayDL1Calib(Tool):
    name = "DisplayDL1Calib"
    description = "Calibrate dl0 data to dl1, and plot the photoelectron " \
                  "images."

    telescope = Int(None, allow_none=True,
                    help='Telescope to view. Set to None to display all '
                         'telescopes.').tag(config=True)

    aliases = Dict(dict(f='EventFileReaderFactory.input_path',
                        r='EventFileReaderFactory.reader',
                        max_events='EventFileReaderFactory.max_events',
                        extractor='ChargeExtractorFactory.extractor',
                        window_width='ChargeExtractorFactory.window_width',
                        t0='ChargeExtractorFactory.t0',
                        window_shift='ChargeExtractorFactory.window_shift',
                        sig_amp_cut_HG='ChargeExtractorFactory.sig_amp_cut_HG',
                        sig_amp_cut_LG='ChargeExtractorFactory.sig_amp_cut_LG',
                        lwt='ChargeExtractorFactory.lwt',
                        clip_amplitude='CameraDL1Calibrator.clip_amplitude',
                        T='DisplayDL1Calib.telescope',
                        O='ImagePlotter.output_path'
                        ))
    flags = Dict(dict(D=({'ImagePlotter': {'display': True}},
                         "Display the photoelectron images on-screen as they "
                         "are produced.")
                      ))
    classes = List([EventFileReaderFactory,
                    ChargeExtractorFactory,
                    CameraDL1Calibrator,
                    ImagePlotter
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reader = None
        self.calibrator = None
        self.plotter = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.reader = reader_class(**kwargs)

        self.calibrator = CameraCalibrator(origin=self.reader.origin, **kwargs)

        self.plotter = ImagePlotter(**kwargs)

    def start(self):
        source = self.reader.read()
        for event in source:
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

if __name__ == '__main__':
    exe = DisplayDL1Calib()
    exe.run()
