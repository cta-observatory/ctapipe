#!/usr/bin/env python3
"""
This example uses a configuration file in JSON format to
process the events and apply pre-selection cuts to the images
(charge and number of pixels).
An HDF5 file is written with image MC and moment parameters
(e.g. length, width, image amplitude, etc.).
"""

import numpy as np
from tqdm import tqdm

from ctapipe.core import Tool
from ctapipe.core.traits import Unicode, List, Dict, Bool
from ctapipe.io import EventSource, HDF5TableWriter

from ctapipe.calib import CameraCalibrator
from ctapipe.utils.CutFlow import CutFlow
from ctapipe.image import hillas_parameters, tailcuts_clean


class SimpleEventWriter(Tool):
    name = 'ctapipe-simple-event-writer'
    description = Unicode(__doc__)

    infile = Unicode(help='input file to read', default='').tag(config=True)
    outfile = Unicode(help='output file name', default_value='output.h5').tag(config=True)
    progress = Bool(help='display progress bar', default_value=True).tag(config=True)

    aliases = Dict({
        'infile': 'EventSource.input_url',
        'outfile': 'SimpleEventWriter.outfile',
        'max-events': 'EventSource.max_events',
        'progress': 'SimpleEventWriter.progress'
    })
    classes = List([EventSource, CameraCalibrator, CutFlow])

    def setup(self):
        self.log.info('Configure EventSource...')

        self.event_source = EventSource.from_config(
            config=self.config,
            parent=self
        )
        self.event_source.allowed_tels = self.config['Analysis']['allowed_tels']

        self.calibrator = CameraCalibrator(
            config=self.config, parent=self, eventsource=self.event_source
        )

        self.writer = HDF5TableWriter(
            filename=self.outfile, group_name='image_infos', overwrite=True
        )

        # Define Pre-selection for images
        preselcuts = self.config['Preselect']
        self.image_cutflow = CutFlow('Image preselection')
        self.image_cutflow.set_cuts(dict(
            no_sel=None,
            n_pixel=lambda s: np.count_nonzero(s) < preselcuts['n_pixel']['min'],
            image_amplitude=lambda q: q < preselcuts['image_amplitude']['min']
        ))

        # Define Pre-selection for events
        self.event_cutflow = CutFlow('Event preselection')
        self.event_cutflow.set_cuts(dict(
            no_sel=None
        ))

    def start(self):
        self.log.info('Loop on events...')

        for event in tqdm(
                self.event_source,
                desc='EventWriter',
                total=self.event_source.max_events,
                disable=~self.progress):

            self.event_cutflow.count('no_sel')
            self.calibrator.calibrate(event)

            for tel_id in event.dl0.tels_with_data:
                self.image_cutflow.count('no_sel')

                camera = event.inst.subarray.tel[tel_id].camera
                dl1_tel = event.dl1.tel[tel_id]

                # Image cleaning
                image = dl1_tel.image[0]  # Waiting for automatic gain selection
                mask = tailcuts_clean(camera, image, picture_thresh=10, boundary_thresh=5)
                cleaned = image.copy()
                cleaned[~mask] = 0

                # Preselection cuts
                if self.image_cutflow.cut('n_pixel', cleaned):
                    continue
                if self.image_cutflow.cut('image_amplitude', np.sum(cleaned)):
                    continue

                # Image parametrisation
                params = hillas_parameters(camera, cleaned)

                # Save Ids, MC infos and Hillas informations
                self.writer.write(camera.cam_id, [event.r0, event.mc, params])

    def finish(self):
        self.log.info('End of job.')

        self.image_cutflow()
        self.event_cutflow()
        self.writer.close()


if __name__ == '__main__':
    tool = SimpleEventWriter()
    tool.run()

