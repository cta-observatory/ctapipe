from ctapipe.core import Component
from ctapipe.plotting.camera import CameraPlotter
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from traitlets import Bool
from traitlets import Unicode



class DisplayTelescopeStep(Component):
    """DisplayTelescopeStep` class represents a Stage for pipeline.
        it Display RawCameraData with matplotlib
    """
    display = Bool(default_value=False, help='Display the camera events') \
     .tag(config=True)
    pdf_path = Unicode(default_value=None,allow_none=True,
        help='Path to store a pdf output of the plots').tag(config=True)


    def init(self):
        self.log.info("--- DisplayTelescopeStep init ---")
        self.fig = plt.figure(figsize=(16, 7))
        self.pdfPages = None
        if self.pdf_path:
            self.pdfPages = PdfPages(self.pdf_path)
        return True

    def run(self,parameters):
        calibrated_event,  geom_dict = parameters
        for tel_id in calibrated_event.dl0.tels_with_data:
            self.fig.clear()
            cam_dimensions = (calibrated_event.dl0.tel[tel_id].num_pixels,
                              calibrated_event.meta.optical_foclen[tel_id])
            self.fig.suptitle("EVENT {} {:.1e} @({:.1f},{:.1f}) @{:.1f}"
                         .format(calibrated_event.dl1.event_id, calibrated_event.mc.energy,
                                 calibrated_event.mc.alt,
                                 calibrated_event.mc.az,
                                 np.sqrt(pow(calibrated_event.mc.core_x, 2) +
                                         pow(calibrated_event.mc.core_y, 2))))
            # Select number of pads to display (will depend on the integrator):
            # charge and/or time of maximum.
            # This last one is displayed only if the integrator calculates it
            npads = 1
            if not calibrated_event.dl1.tel[tel_id].peakpos[0] is None:
                npads = 2
            # Only create two pads if there is timing information extracted
            # from the calibration
            ax1 = self.fig.add_subplot(1, npads, 1)
            # If the geometery has not already been added to geom_dict, it will
            # be added in CameraPlotter
            plotter = CameraPlotter(calibrated_event, geom_dict)
            signals = calibrated_event.dl1.tel[tel_id].pe_charge
            camera1 = plotter.draw_camera(tel_id, signals, ax1)
            cmaxmin = (max(signals) - min(signals))
            color_list = [(0 / cmaxmin, 'darkblue'),
                       (np.abs(min(signals)) / cmaxmin, 'black'),
                       (2.0 * np.abs(min(signals)) / cmaxmin, 'blue'),
                       (2.5 * np.abs(min(signals)) / cmaxmin, 'green'),
                       (1, 'yellow')]
            try:
                cmap_charge = colors.LinearSegmentedColormap.from_list('cmap_c',
                    color_list)
                camera1.pixels.set_cmap(cmap_charge)
                camera1.add_colorbar(ax=ax1, label=" [photo-electrons]")
            except:
                camera1.pixels.set_cmap('jet')
            ax1.set_title("CT {} ({}) - Mean pixel charge"
                          .format(tel_id, geom_dict[cam_dimensions].cam_id))
            if not calibrated_event.dl1.tel[tel_id].peakpos[0] is None:
                ax2 = self.fig.add_subplot(1, npads, npads)
                times = calibrated_event.dl1.tel[tel_id].peakpos
                camera2 = plotter.draw_camera(tel_id, times, ax2)
                tmaxmin = calibrated_event.dl0.tel[tel_id].num_samples
                t_chargemax = times[signals.argmax()]
                if t_chargemax > 15:
                    t_chargemax = 7
                cmap_time = colors.LinearSegmentedColormap.from_list(
                    'cmap_t', [(0 / tmaxmin, 'darkgreen'),
                               (0.6 * t_chargemax / tmaxmin, 'green'),
                               (t_chargemax / tmaxmin, 'yellow'),
                               (1.4 * t_chargemax / tmaxmin, 'blue'),
                               (1, 'darkblue')])
                try:
                    camera2.pixels.set_cmap(cmap_time)
                    camera2.add_colorbar(ax=ax2, label="[time slice]")
                except:
                    camera2.pixels.set_cmap('jet')
                ax2.set_title("CT {} ({}) - Pixel peak position"
                              .format(tel_id, geom_dict[cam_dimensions].cam_id))
            if self.display:
                plt.pause(.1)
            if self.pdfPages is not None:
                self.pdfPages.savefig(self.fig)

    def finish(self):
        self.log.info("--- DisplayTelescopeStep finish ---")
        if self.pdfPages:
            self.pdfPages.close()
        pass
