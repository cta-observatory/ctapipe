"""
Create a plot of where the integration window lays on the trace for the pixel
with the highest charge, its neighbours, and the pixel with the lowest max
charge and its neighbours. Also shows a disgram of which pixels count as a
neighbour, the camera image for the max charge timeslice, the true pe camera
image, and a calibrated camera image
"""

import os

import numpy as np
from matplotlib import pyplot as plt
from traitlets import Dict, List, Int, Bool, Unicode, Enum

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.core import Tool, Component
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from ctapipe.instrument import CameraGeometry
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.visualization import CameraDisplay


class IntegratorPlotter(Component):
    name = 'IntegratorPlotter'

    output_dir = Unicode(None, allow_none=True,
                         help='Output path to the directory where the plots '
                              'will be saved. If None, a directory is created '
                              'in the location of the '
                              'input file.').tag(config=True)

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
        self._init_figure()

    def _init_figure(self):
        self.fig = plt.figure(figsize=(16, 7))
        self.ax_intensity = self.fig.add_subplot(1, 2, 1)
        self.ax_peakpos = self.fig.add_subplot(1, 2, 2)

    def plot(self, input_file, event, telid, chan, extractor_name):
        # Extract required images
        dl0 = event.dl0.tel[telid].pe_samples[chan]

        t_pe = event.mc.tel[telid].photo_electron_image
        dl1 = event.dl1.tel[telid].image[chan]
        max_time = np.unravel_index(np.argmax(dl0), dl0.shape)[1]
        max_charges = np.max(dl0, axis=1)
        max_pix = int(np.argmax(max_charges))
        min_pix = int(np.argmin(max_charges))

        geom = CameraGeometry.guess(*event.inst.pixel_pos[telid],
                                    event.inst.optical_foclen[telid])

        nei = geom.neighbors

        # Get Neighbours
        max_pixel_nei = nei[max_pix]
        min_pixel_nei = nei[min_pix]

        # Get Windows
        windows = event.dl1.tel[telid].extracted_samples[chan]
        length = np.sum(windows, axis=1)
        start = np.argmax(windows, axis=1)
        end = start + length - 1

        # Draw figures
        ax_max_nei = {}
        ax_min_nei = {}
        fig_waveforms = plt.figure(figsize=(18, 9))
        fig_waveforms.subplots_adjust(hspace=.5)
        fig_camera = plt.figure(figsize=(15, 12))

        ax_max_pix = fig_waveforms.add_subplot(4, 2, 1)
        ax_min_pix = fig_waveforms.add_subplot(4, 2, 2)
        ax_max_nei[0] = fig_waveforms.add_subplot(4, 2, 3)
        ax_min_nei[0] = fig_waveforms.add_subplot(4, 2, 4)
        ax_max_nei[1] = fig_waveforms.add_subplot(4, 2, 5)
        ax_min_nei[1] = fig_waveforms.add_subplot(4, 2, 6)
        ax_max_nei[2] = fig_waveforms.add_subplot(4, 2, 7)
        ax_min_nei[2] = fig_waveforms.add_subplot(4, 2, 8)

        ax_img_nei = fig_camera.add_subplot(2, 2, 1)
        ax_img_max = fig_camera.add_subplot(2, 2, 2)
        ax_img_true = fig_camera.add_subplot(2, 2, 3)
        ax_img_cal = fig_camera.add_subplot(2, 2, 4)

        # Draw max pixel traces
        ax_max_pix.plot(dl0[max_pix])
        ax_max_pix.set_xlabel("Time (ns)")
        ax_max_pix.set_ylabel("DL0 Samples (ADC)")
        ax_max_pix.set_title("(Max) Pixel: {}, True: {}, Measured = {:.3f}"
                             .format(max_pix, t_pe[max_pix], dl1[max_pix]))
        max_ylim = ax_max_pix.get_ylim()
        ax_max_pix.plot([start[max_pix], start[max_pix]],
                        ax_max_pix.get_ylim(), color='r', alpha=1)
        ax_max_pix.plot([end[max_pix], end[max_pix]],
                        ax_max_pix.get_ylim(), color='r', alpha=1)
        for i, ax in ax_max_nei.items():
            if len(max_pixel_nei) > i:
                pix = max_pixel_nei[i]
                ax.plot(dl0[pix])
                ax.set_xlabel("Time (ns)")
                ax.set_ylabel("DL0 Samples (ADC)")
                ax.set_title("(Max Nei) Pixel: {}, True: {}, Measured = {:.3f}"
                             .format(pix, t_pe[pix], dl1[pix]))
                ax.set_ylim(max_ylim)
                ax.plot([start[pix], start[pix]],
                        ax.get_ylim(), color='r', alpha=1)
                ax.plot([end[pix], end[pix]],
                        ax.get_ylim(), color='r', alpha=1)

        # Draw min pixel traces
        ax_min_pix.plot(dl0[min_pix])
        ax_min_pix.set_xlabel("Time (ns)")
        ax_min_pix.set_ylabel("DL0 Samples (ADC)")
        ax_min_pix.set_title("(Min) Pixel: {}, True: {}, Measured = {:.3f}"
                             .format(min_pix, t_pe[min_pix], dl1[min_pix]))
        ax_min_pix.set_ylim(max_ylim)
        ax_min_pix.plot([start[min_pix], start[min_pix]],
                        ax_min_pix.get_ylim(), color='r', alpha=1)
        ax_min_pix.plot([end[min_pix], end[min_pix]],
                        ax_min_pix.get_ylim(), color='r', alpha=1)
        for i, ax in ax_min_nei.items():
            if len(min_pixel_nei) > i:
                pix = min_pixel_nei[i]
                ax.plot(dl0[pix])
                ax.set_xlabel("Time (ns)")
                ax.set_ylabel("DL0 Samples (ADC)")
                ax.set_title("(Min Nei) Pixel: {}, True: {}, Measured = {:.3f}"
                             .format(pix, t_pe[pix], dl1[pix]))
                ax.set_ylim(max_ylim)
                ax.plot([start[pix], start[pix]],
                        ax.get_ylim(), color='r', alpha=1)
                ax.plot([end[pix], end[pix]],
                        ax.get_ylim(), color='r', alpha=1)

        # Draw cameras
        nei_camera = np.zeros_like(max_charges, dtype=np.int)
        nei_camera[min_pixel_nei] = 2
        nei_camera[min_pix] = 1
        nei_camera[max_pixel_nei] = 3
        nei_camera[max_pix] = 4
        camera = CameraDisplay(geom, ax=ax_img_nei)
        camera.image = nei_camera
        camera.cmap = plt.cm.viridis
        ax_img_nei.set_title("Neighbour Map")
        ax_img_nei.annotate("Pixel: {}".format(max_pix),
                            xy=(geom.pix_x.value[max_pix],
                                geom.pix_y.value[max_pix]),
                            xycoords='data', xytext=(0.05, 0.98),
                            textcoords='axes fraction',
                            arrowprops=dict(facecolor='red', width=2,
                                            alpha=0.4),
                            horizontalalignment='left',
                            verticalalignment='top')
        ax_img_nei.annotate("Pixel: {}".format(min_pix),
                            xy=(geom.pix_x.value[min_pix],
                                geom.pix_y.value[min_pix]),
                            xycoords='data', xytext=(0.05, 0.94),
                            textcoords='axes fraction',
                            arrowprops=dict(facecolor='orange', width=2,
                                            alpha=0.4),
                            horizontalalignment='left',
                            verticalalignment='top')
        camera = CameraDisplay(geom, ax=ax_img_max)
        camera.image = dl0[:, max_time]
        camera.cmap = plt.cm.viridis
        camera.add_colorbar(ax=ax_img_max, label="DL0 Samples (ADC)")
        ax_img_max.set_title("Max Timeslice (T = {})".format(max_time))
        ax_img_max.annotate("Pixel: {}".format(max_pix),
                            xy=(geom.pix_x.value[max_pix],
                                geom.pix_y.value[max_pix]),
                            xycoords='data', xytext=(0.05, 0.98),
                            textcoords='axes fraction',
                            arrowprops=dict(facecolor='red', width=2,
                                            alpha=0.4),
                            horizontalalignment='left',
                            verticalalignment='top')
        ax_img_max.annotate("Pixel: {}".format(min_pix),
                            xy=(geom.pix_x.value[min_pix],
                                geom.pix_y.value[min_pix]),
                            xycoords='data', xytext=(0.05, 0.94),
                            textcoords='axes fraction',
                            arrowprops=dict(facecolor='orange', width=2,
                                            alpha=0.4),
                            horizontalalignment='left',
                            verticalalignment='top')

        camera = CameraDisplay(geom, ax=ax_img_true)
        camera.image = t_pe
        camera.cmap = plt.cm.viridis
        camera.add_colorbar(ax=ax_img_true, label="True Charge (p.e.)")
        ax_img_true.set_title("True Charge")
        ax_img_true.annotate("Pixel: {}".format(max_pix),
                             xy=(geom.pix_x.value[max_pix],
                                 geom.pix_y.value[max_pix]),
                             xycoords='data', xytext=(0.05, 0.98),
                             textcoords='axes fraction',
                             arrowprops=dict(facecolor='red', width=2,
                                             alpha=0.4),
                             horizontalalignment='left',
                             verticalalignment='top')
        ax_img_true.annotate("Pixel: {}".format(min_pix),
                             xy=(geom.pix_x.value[min_pix],
                                 geom.pix_y.value[min_pix]),
                             xycoords='data', xytext=(0.05, 0.94),
                             textcoords='axes fraction',
                             arrowprops=dict(facecolor='orange', width=2,
                                             alpha=0.4),
                             horizontalalignment='left',
                             verticalalignment='top')

        camera = CameraDisplay(geom, ax=ax_img_cal)
        camera.image = dl1
        camera.cmap = plt.cm.viridis
        camera.add_colorbar(ax=ax_img_cal,
                            label="Calib Charge (Photo-electrons)")
        ax_img_cal.set_title("Charge (integrator={})".format(extractor_name))
        ax_img_cal.annotate("Pixel: {}".format(max_pix),
                            xy=(geom.pix_x.value[max_pix],
                                geom.pix_y.value[max_pix]),
                            xycoords='data', xytext=(0.05, 0.98),
                            textcoords='axes fraction',
                            arrowprops=dict(facecolor='red', width=2,
                                            alpha=0.4),
                            horizontalalignment='left',
                            verticalalignment='top')
        ax_img_cal.annotate("Pixel: {}".format(min_pix),
                            xy=(geom.pix_x.value[min_pix],
                                geom.pix_y.value[min_pix]),
                            xycoords='data', xytext=(0.05, 0.94),
                            textcoords='axes fraction',
                            arrowprops=dict(facecolor='orange', width=2,
                                            alpha=0.4),
                            horizontalalignment='left',
                            verticalalignment='top')

        fig_waveforms.suptitle("Integrator = {}".format(extractor_name))
        fig_camera.suptitle("Camera = {}".format(geom.cam_id))

        waveform_output_name = "e{}_t{}_c{}_extractor{}_waveform.pdf"\
            .format(event.count, telid, chan, extractor_name)
        camera_output_name = "e{}_t{}_c{}_extractor{}_camera.pdf"\
            .format(event.count, telid, chan, extractor_name)

        output_dir = self.output_dir
        if output_dir is None:
            output_dir = input_file.output_directory
        output_dir = os.path.join(output_dir, self.name)
        if not os.path.exists(output_dir):
            self.log.info("Creating directory: {}".format(output_dir))
            os.makedirs(output_dir)

        waveform_output_path = os.path.join(output_dir, waveform_output_name)
        self.log.info("Saving: {}".format(waveform_output_path))
        fig_waveforms.savefig(waveform_output_path, format='pdf',
                              bbox_inches='tight')

        camera_output_path = os.path.join(output_dir, camera_output_name)
        self.log.info("Saving: {}".format(camera_output_path))
        fig_camera.savefig(camera_output_path, format='pdf',
                           bbox_inches='tight')
        return geom


class DisplayIntegrator(Tool):
    name = "DisplayIntegrator"
    description = "Calibrate dl0 data to dl1, and plot the various camera " \
                  "images that characterise the event and calibration. Also " \
                  "plot some examples of waveforms with the " \
                  "integration window."

    event_index = Int(0, help='Event index to view.').tag(config=True)
    use_event_id = Bool(False, help='event_index will obtain an event using'
                                    'event_id instead of '
                                    'index.').tag(config=True)
    telescope = Int(None, allow_none=True,
                    help='Telescope to view. Set to None to display the first'
                         'telescope with data.').tag(config=True)
    channel = Enum([0, 1], 0, help='Channel to view').tag(config=True)

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        extractor='ChargeExtractorFactory.extractor',
                        window_width='ChargeExtractorFactory.window_width',
                        window_shift='ChargeExtractorFactory.window_shift',
                        sig_amp_cut_HG='ChargeExtractorFactory.sig_amp_cut_HG',
                        sig_amp_cut_LG='ChargeExtractorFactory.sig_amp_cut_LG',
                        lwt='ChargeExtractorFactory.lwt',
                        clip_amplitude='CameraDL1Calibrator.clip_amplitude',
                        radius='CameraDL1Calibrator.radius',
                        E='DisplayIntegrator.event_index',
                        T='DisplayIntegrator.telescope',
                        C='DisplayIntegrator.channel',
                        O='IntegratorPlotter.output_dir'
                        ))
    flags = Dict(dict(id=({'DisplayDL1Calib': {'use_event_index': True}},
                          'event_index will obtain an event using '
                          'event_id instead of index.')
                      ))
    classes = List([EventFileReaderFactory,
                    ChargeExtractorFactory,
                    CameraDL1Calibrator,
                    IntegratorPlotter
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_reader = None
        self.r1 = None
        self.dl0 = None
        self.extractor = None
        self.dl1 = None
        self.plotter = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.file_reader = reader_class(**kwargs)

        extractor_factory = ChargeExtractorFactory(**kwargs)
        extractor_class = extractor_factory.get_class()
        self.extractor = extractor_class(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin=self.file_reader.origin,
                                               **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(extractor=self.extractor, **kwargs)

        self.plotter = IntegratorPlotter(**kwargs)

    def start(self):
        event = self.file_reader.get_event(self.event_index, self.use_event_id)

        # Calibrate
        self.r1.calibrate(event)
        self.dl0.reduce(event)
        self.dl1.calibrate(event)

        # Select telescope
        tels = list(event.r0.tels_with_data)
        telid = self.telescope
        if telid is None:
            telid = tels[0]
        if telid not in tels:
            self.log.error("[event] please specify one of the following "
                           "telescopes for this event: {}".format(tels))
            exit()

        extractor_name = self.extractor.name

        self.plotter.plot(self.file_reader, event, telid,
                          self.channel, extractor_name)

    def finish(self):
        pass

if __name__ == '__main__':
    exe = DisplayIntegrator()
    exe.run()
