#!/usr/bin/env python3

"""
Example of extracting data for single telescope from a merged/interleaved
simtelarray data file and displaying it.

Only events that contain the specified telescope are read and
displayed. Other telescopes and events are skipped over (EventIO data
files have no index table in them, so the events must be read in
sequence to find ones with the appropriate telescope, therefore this
is not a fast operation)
"""

import argparse
import logging

import numpy as np
from ctapipe.instrument import CameraGeometry
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import get_dataset
from ctapipe.visualization import CameraDisplay
from ctapipe.calib import CameraCalibrator
from ctapipe.image import tailcuts_clean, hillas_parameters
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from ctapipe.core import Tool, ToolConfigurationError
from ctapipe.core.traits import *
from ctapipe.io.eventfilereader import EventFileReaderFactory

logging.basicConfig(level=logging.DEBUG)


class SingleTelEventDisplay(Tool):
    name="ctapipe-display-single-tel"
    description=Unicode(__doc__)
    
    infile = Unicode(help="input file to read", default='').tag(config=True)
    tel = Int(help='Telescope ID to display', default=0).tag(config=True)
    channel = Integer(help="channel number to display", min=0, max=1).tag(config=True)
    write = Bool(help="Write out images to PNG files", default=False).tag(
        config=True)
    clean = Bool(help="Apply image cleaning", default=False).tag(config=True)
    hillas = Bool(help="Apply and display Hillas parametrization", 
                  default=False).tag(config=True)
    samples =Bool(help="Show each sample", default=False).tag(config=True)
    
    aliases = Dict({'infile': 'EventFileReaderFactory.input_path',
                    'tel':'SingleTelEventDisplay.tel',
                    'max-events': 'EventFileReaderFactory.max_events',
                    'channel' : 'SingleTelEventDisplay.channel',
                    'write' : 'SingleTelEventDisplay.write',
                    'clean' : 'SingleTelEventDisplay.clean',
                    'hillas' : 'SingleTelEventDisplay.hillas',
                    'samples' : 'SingleTelEventDisplay.samples'})

    classes =  List([EventFileReaderFactory, CameraCalibrator])
    
    def setup(self):

        reader_factory = EventFileReaderFactory(None, self)
        reader_class = reader_factory.get_class()
        self.reader = reader_class(None,self)

        self.calibrator = CameraCalibrator(config=None, tool=self,
                                           origin=self.reader.origin)


#        if self.infile == '':
#            raise ToolConfigurationError("Need to specify --infile <filename>")

        self.calib = CameraCalibrator(config=None, tool=self)

        self.source = self.reader.read(allowed_tels=[self.tel, ])

        self.log.info('SELECTING EVENTS FROM TELESCOPE {}'.format(self.tel))


    
    def start(self):

        disp = None

        for event in self.source:

            self.log.info('Scanning input file... count = {}'.format(event.count))
            self.log.debug(event.trig)
            self.log.debug(event.mc)
            self.log.debug(event.dl0)

            self.calib.calibrate(event)

            if disp is None:
                x, y = event.inst.pixel_pos[self.tel]
                focal_len = event.inst.optical_foclen[self.tel]
                geom = CameraGeometry.guess(x, y, focal_len)
                self.log.info(geom.pix_x)
                disp = CameraDisplay(geom)
                # disp.enable_pixel_picker()
                disp.add_colorbar()
                plt.show(block=False)

            # display the event
            disp.axes.set_title('CT{:03d} ({}), event {:010d}'.format(
                self.tel, geom.cam_id, event.r0.event_id)
            )

            if self.samples:
                # display time-varying event
                data = event.dl0.tel[self.tel].pe_samples[self.channel]
                for ii in range(data.shape[1]):
                    disp.image = data[:, ii]
                    disp.set_limits_percent(70)
                    plt.suptitle("Sample {:03d}".format(ii))
                    plt.pause(0.01)
                    if self.write:
                        plt.savefig('CT{:03d}_EV{:010d}_S{:02d}.png'
                                    .format(self.tel, event.r0.event_id, ii))
            else:
                # display integrated event:
                im = event.dl1.tel[self.tel].image[self.channel]

                if self.clean:
                    mask = tailcuts_clean(geom, im, picture_thresh=10,
                                          boundary_thresh=7)
                    im[~mask] = 0.0

                disp.image = im

                if self.hillas:
                    params = hillas_parameters(pix_x=geom.pix_x,
                                               pix_y=geom.pix_y, image=im)
                    ellipses = disp.axes.findobj(Ellipse)
                    if len(ellipses) > 0:
                        ellipses[0].remove()
                    disp.overlay_moments(params, color='pink', lw=3,
                                         with_label=False)

                plt.pause(1.0)
                if self.write:
                    plt.savefig('CT{:03d}_EV{:010d}.png'
                                .format(self.tel, event.r0.event_id))

        self.log.info("FINISHED READING DATA FILE")

        if disp is None:
            self.log.warning('No events for tel {} were found in {}. Try a '
                             'different EventIO file or another telescope'
                             .format(self.tel, self.infile),
                             )

        pass


if __name__ == '__main__':

    tool = SingleTelEventDisplay()
    tool.run()
