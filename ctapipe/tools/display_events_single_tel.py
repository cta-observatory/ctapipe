#!/usr/bin/env python3
"""
Loops over events in a data file and displays them, with optional image
cleaning and hillas parameter overlays.

Only events that contain the specified telescope are read and
displayed. Other telescopes and events are skipped over (EventIO data
files have no index table in them, so the events must be read in
sequence to find ones with the appropriate telescope, therefore this
is not a fast operation)
"""

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm

from ctapipe.calib import CameraCalibrator
from ctapipe.core import Tool
from ctapipe.core.traits import Float, Dict, List
from ctapipe.core.traits import Unicode, Int, Bool
from ctapipe.image import (
    tailcuts_clean, hillas_parameters, HillasParameterizationError
)
from ctapipe.io import EventSource
from ctapipe.visualization import CameraDisplay


class SingleTelEventDisplay(Tool):
    name = "ctapipe-display-televents"
    description = Unicode(__doc__)

    infile = Unicode(help="input file to read", default='').tag(config=True)
    tel = Int(help='Telescope ID to display', default=0).tag(config=True)
    write = Bool(
        help="Write out images to PNG files", default=False
    ).tag(config=True)
    clean = Bool(help="Apply image cleaning", default=False).tag(config=True)
    hillas = Bool(
        help="Apply and display Hillas parametrization", default=False
    ).tag(config=True)
    samples = Bool(help="Show each sample", default=False).tag(config=True)
    display = Bool(
        help="Display results in interactive window", default_value=True
    ).tag(config=True)
    delay = Float(
        help='delay between events in s', default_value=0.01, min=0.001
    ).tag(config=True)
    progress = Bool(
        help='display progress bar', default_value=True
    ).tag(config=True)

    aliases = Dict({
        'infile': 'SingleTelEventDisplay.infile',
        'tel': 'SingleTelEventDisplay.tel',
        'max-events': 'EventSource.max_events',
        'write': 'SingleTelEventDisplay.write',
        'clean': 'SingleTelEventDisplay.clean',
        'hillas': 'SingleTelEventDisplay.hillas',
        'samples': 'SingleTelEventDisplay.samples',
        'display': 'SingleTelEventDisplay.display',
        'delay': 'SingleTelEventDisplay.delay',
        'progress': 'SingleTelEventDisplay.progress'
    })

    classes = List([EventSource, CameraCalibrator])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):
        print('TOLLES INFILE', self.infile)
        self.event_source = self.add_component(
            EventSource.from_url(self.infile, parent=self)
        )
        self.event_source.allowed_tels = {self.tel, }

        self.calibrator = self.add_component(
            CameraCalibrator(parent=self, subarray=self.event_source.subarray)
        )

        self.log.info(f'SELECTING EVENTS FROM TELESCOPE {self.tel}')

    def start(self):

        disp = None

        for event in tqdm(
                self.event_source,
                desc=f'Tel{self.tel}',
                total=self.event_source.max_events,
                disable=~self.progress
        ):

            self.log.debug(event.trig)
            self.log.debug(f"Energy: {event.mc.energy}")

            self.calibrator(event)

            if disp is None:
                geom = event.inst.subarray.tel[self.tel].camera
                self.log.info(geom)
                disp = CameraDisplay(geom)
                # disp.enable_pixel_picker()
                disp.add_colorbar()
                if self.display:
                    plt.show(block=False)

            # display the event
            disp.axes.set_title(
                'CT{:03d} ({}), event {:06d}'.format(
                    self.tel, geom.cam_id, event.r0.event_id
                )
            )

            if self.samples:
                # display time-varying event
                data = event.dl0.tel[self.tel].waveform
                for ii in range(data.shape[1]):
                    disp.image = data[:, ii]
                    disp.set_limits_percent(70)
                    plt.suptitle(f"Sample {ii:03d}")
                    if self.display:
                        plt.pause(self.delay)
                    if self.write:
                        plt.savefig(
                            f'CT{self.tel:03d}_EV{event.r0.event_id:10d}'
                            f'_S{ii:02d}.png'
                        )
            else:
                # display integrated event:
                im = event.dl1.tel[self.tel].image

                if self.clean:
                    mask = tailcuts_clean(
                        geom, im, picture_thresh=10, boundary_thresh=7
                    )
                    im[~mask] = 0.0

                disp.image = im

                if self.hillas:
                    try:
                        ellipses = disp.axes.findobj(Ellipse)
                        if len(ellipses) > 0:
                            ellipses[0].remove()

                        params = hillas_parameters(geom, image=im)
                        disp.overlay_moments(
                            params, color='pink', lw=3, with_label=False
                        )
                    except HillasParameterizationError:
                        pass

                if self.display:
                    plt.pause(self.delay)
                if self.write:
                    plt.savefig(
                        f'CT{self.tel:03d}_EV{event.r0.event_id:010d}.png'
                    )

        self.log.info("FINISHED READING DATA FILE")

        if disp is None:
            self.log.warning(
                'No events for tel {} were found in {}. Try a '
                'different EventIO file or another telescope'
                    .format(self.tel, self.infile),
            )


def main():
    tool = SingleTelEventDisplay()
    tool.run()
