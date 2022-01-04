"""
Calibrate dl0 data to dl1, and plot the various camera images that
characterise the event and calibration. Also plot some examples of waveforms
with the integration window.
"""
import numpy as np
from matplotlib import pyplot as plt

from ..calib import CameraCalibrator
from ..core import Tool
from ..core.traits import Int, Bool, Enum, flag, classes_with_traits
from ..io import EventSource
from ..io.eventseeker import EventSeeker
from ..visualization import CameraDisplay


def plot(subarray, event, telid, chan, extractor_name):
    # Extract required images
    dl0 = event.dl0.tel[telid].waveform

    dl1 = event.dl1.tel[telid].image
    t_pe = event.simulation.tel[telid].true_image
    if t_pe is None:
        t_pe = np.zeros_like(dl1)
    max_time = np.unravel_index(np.argmax(dl0), dl0.shape)[1]
    max_charges = np.max(dl0, axis=1)
    max_pix = int(np.argmax(max_charges))
    min_pix = int(np.argmin(max_charges))

    geom = subarray.tel[telid].camera.geometry
    nei = geom.neighbors

    # Get Neighbours
    max_pixel_nei = nei[max_pix]
    min_pixel_nei = nei[min_pix]

    # Draw figures
    ax_max_nei = {}
    ax_min_nei = {}
    fig_waveforms = plt.figure(figsize=(18, 9))
    fig_waveforms.subplots_adjust(hspace=0.5)
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
    ax_max_pix.set_title(
        f"(Max) Pixel: {max_pix}, True: {t_pe[max_pix]}, "
        f"Measured = {dl1[max_pix]:.3f}"
    )
    max_ylim = ax_max_pix.get_ylim()
    for i, ax in ax_max_nei.items():
        if len(max_pixel_nei) > i:
            pix = max_pixel_nei[i]
            ax.plot(dl0[pix])
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("DL0 Samples (ADC)")
            ax.set_title(
                "(Max Nei) Pixel: {}, True: {}, Measured = {:.3f}".format(
                    pix, t_pe[pix], dl1[pix]
                )
            )
            ax.set_ylim(max_ylim)

    # Draw min pixel traces
    ax_min_pix.plot(dl0[min_pix])
    ax_min_pix.set_xlabel("Time (ns)")
    ax_min_pix.set_ylabel("DL0 Samples (ADC)")
    ax_min_pix.set_title(
        f"(Min) Pixel: {min_pix}, True: {t_pe[min_pix]}, "
        f"Measured = {dl1[min_pix]:.3f}"
    )
    ax_min_pix.set_ylim(max_ylim)
    for i, ax in ax_min_nei.items():
        if len(min_pixel_nei) > i:
            pix = min_pixel_nei[i]
            ax.plot(dl0[pix])
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("DL0 Samples (ADC)")
            ax.set_title(
                f"(Min Nei) Pixel: {pix}, True: {t_pe[pix]}, "
                f"Measured = {dl1[pix]:.3f}"
            )
            ax.set_ylim(max_ylim)

    # Draw cameras
    nei_camera = np.zeros_like(max_charges, dtype=np.int64)
    nei_camera[min_pixel_nei] = 2
    nei_camera[min_pix] = 1
    nei_camera[max_pixel_nei] = 3
    nei_camera[max_pix] = 4
    camera = CameraDisplay(geom, ax=ax_img_nei)
    camera.image = nei_camera
    ax_img_nei.set_title("Neighbour Map")
    ax_img_nei.annotate(
        f"Pixel: {max_pix}",
        xy=(geom.pix_x.value[max_pix], geom.pix_y.value[max_pix]),
        xycoords="data",
        xytext=(0.05, 0.98),
        textcoords="axes fraction",
        arrowprops=dict(facecolor="red", width=2, alpha=0.4),
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax_img_nei.annotate(
        f"Pixel: {min_pix}",
        xy=(geom.pix_x.value[min_pix], geom.pix_y.value[min_pix]),
        xycoords="data",
        xytext=(0.05, 0.94),
        textcoords="axes fraction",
        arrowprops=dict(facecolor="orange", width=2, alpha=0.4),
        horizontalalignment="left",
        verticalalignment="top",
    )
    camera = CameraDisplay(geom, ax=ax_img_max)
    camera.image = dl0[:, max_time]
    camera.add_colorbar(ax=ax_img_max, label="DL0 Samples (ADC)")
    ax_img_max.set_title(f"Max Timeslice (T = {max_time})")
    ax_img_max.annotate(
        f"Pixel: {max_pix}",
        xy=(geom.pix_x.value[max_pix], geom.pix_y.value[max_pix]),
        xycoords="data",
        xytext=(0.05, 0.98),
        textcoords="axes fraction",
        arrowprops=dict(facecolor="red", width=2, alpha=0.4),
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax_img_max.annotate(
        f"Pixel: {min_pix}",
        xy=(geom.pix_x.value[min_pix], geom.pix_y.value[min_pix]),
        xycoords="data",
        xytext=(0.05, 0.94),
        textcoords="axes fraction",
        arrowprops=dict(facecolor="orange", width=2, alpha=0.4),
        horizontalalignment="left",
        verticalalignment="top",
    )

    camera = CameraDisplay(geom, ax=ax_img_true)
    camera.image = t_pe
    camera.add_colorbar(ax=ax_img_true, label="True Charge (p.e.)")
    ax_img_true.set_title("True Charge")
    ax_img_true.annotate(
        f"Pixel: {max_pix}",
        xy=(geom.pix_x.value[max_pix], geom.pix_y.value[max_pix]),
        xycoords="data",
        xytext=(0.05, 0.98),
        textcoords="axes fraction",
        arrowprops=dict(facecolor="red", width=2, alpha=0.4),
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax_img_true.annotate(
        f"Pixel: {min_pix}",
        xy=(geom.pix_x.value[min_pix], geom.pix_y.value[min_pix]),
        xycoords="data",
        xytext=(0.05, 0.94),
        textcoords="axes fraction",
        arrowprops=dict(facecolor="orange", width=2, alpha=0.4),
        horizontalalignment="left",
        verticalalignment="top",
    )

    camera = CameraDisplay(geom, ax=ax_img_cal)
    camera.image = dl1
    camera.add_colorbar(ax=ax_img_cal, label="Calib Charge (Photo-electrons)")
    ax_img_cal.set_title(f"Charge (integrator={extractor_name})")
    ax_img_cal.annotate(
        f"Pixel: {max_pix}",
        xy=(geom.pix_x.value[max_pix], geom.pix_y.value[max_pix]),
        xycoords="data",
        xytext=(0.05, 0.98),
        textcoords="axes fraction",
        arrowprops=dict(facecolor="red", width=2, alpha=0.4),
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax_img_cal.annotate(
        f"Pixel: {min_pix}",
        xy=(geom.pix_x.value[min_pix], geom.pix_y.value[min_pix]),
        xycoords="data",
        xytext=(0.05, 0.94),
        textcoords="axes fraction",
        arrowprops=dict(facecolor="orange", width=2, alpha=0.4),
        horizontalalignment="left",
        verticalalignment="top",
    )

    fig_waveforms.suptitle(f"Integrator = {extractor_name}")
    fig_camera.suptitle(f"Camera = {geom.camera_name}")

    plt.show()


class DisplayIntegrator(Tool):
    name = "ctapipe-display-integration"
    description = __doc__

    event_index = Int(0, help="Event index to view.").tag(config=True)
    use_event_id = Bool(
        False, help="event_index will obtain an event using event_id instead of index."
    ).tag(config=True)
    telescope = Int(
        None,
        allow_none=True,
        help="Telescope to view. Set to None to display the first"
        "telescope with data.",
    ).tag(config=True)
    channel = Enum([0, 1], 0, help="Channel to view").tag(config=True)

    aliases = {
        ("i", "input"): "EventSource.input_url",
        ("m", "max-events"): "EventSource.max_events",
        ("e", "event-index"): "DisplayIntegrator.event_index",
        ("t", "telescope"): "DisplayIntegrator.telescope",
        ("C", "channel"): "DisplayIntegrator.channel",
    }
    flags = flag(
        "id",
        "DisplayDL1Calib.use_event_index",
        "event_index will obtain an event using event_id instead of index.",
        "event_index will obtain an event using index.",
    )
    classes = classes_with_traits(EventSource)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # make sure gzip files are seekable
        self.config.SimTelEventSource.back_seekable = True
        self.eventseeker = None
        self.calibrator = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"

        event_source = EventSource(parent=self)
        self.subarray = event_source.subarray
        self.eventseeker = EventSeeker(event_source, parent=self)
        self.calibrate = CameraCalibrator(parent=self, subarray=self.subarray)

    def start(self):
        if self.use_event_id:
            event = self.eventseeker.get_event_id(self.event_index)
        else:
            event = self.eventseeker.get_event_index(self.event_index)

        # Calibrate
        self.calibrate(event)

        # Select telescope
        tels = list(event.r0.tel.keys())
        telid = self.telescope
        if telid is None:
            telid = tels[0]
        if telid not in tels:
            self.log.error(
                "[event] please specify one of the following "
                "telescopes for this event: {}".format(tels)
            )
            exit()

        extractor_name = self.calibrate.image_extractor_type.tel[telid]

        plot(self.subarray, event, telid, self.channel, extractor_name)

    def finish(self):
        pass


def main():
    exe = DisplayIntegrator()
    exe.run()
