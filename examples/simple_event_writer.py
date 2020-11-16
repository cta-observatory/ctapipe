#!/usr/bin/env python3
"""
This example uses a configuration file in JSON format to
process the events and apply pre-selection cuts to the images
(charge and number of pixels).
An HDF5 file is written with image MC and moment parameters
(e.g. length, width, image amplitude, etc.).
"""
from tqdm import tqdm

from ctapipe.core import Tool
from ctapipe.core.traits import Path, Unicode, List, Dict, Bool
from ctapipe.io import EventSource, HDF5TableWriter

from ctapipe.calib import CameraCalibrator
from ctapipe.utils import get_dataset_path
from ctapipe.image import hillas_parameters, tailcuts_clean


class SimpleEventWriter(Tool):
    name = "ctapipe-simple-event-writer"
    description = Unicode(__doc__)

    output = Path(
        help="output file name", directory_ok=False, default_value="output.h5"
    ).tag(config=True)

    progress = Bool(help="display progress bar", default_value=True).tag(config=True)

    aliases = Dict(
        {
            "input": "EventSource.input_url",
            "output": "SimpleEventWriter.outfile",
            "max-events": "EventSource.max_events",
            "progress": "SimpleEventWriter.progress",
        }
    )
    classes = List([EventSource, CameraCalibrator])

    def setup(self):
        self.log.info("Configure EventSource...")

        EventSource.input_url.default_value = get_dataset_path(
            "lst_prod3_calibration_and_mcphotons.simtel.zst"
        )
        self.event_source = EventSource(parent=self)

        self.calibrator = CameraCalibrator(
            subarray=self.event_source.subarray, parent=self
        )
        self.writer = HDF5TableWriter(
            filename=self.output, group_name="image_infos", overwrite=True, parent=self
        )

    def start(self):
        self.log.info("Loop on events...")

        for event in tqdm(
            self.event_source,
            desc="EventWriter",
            total=self.event_source.max_events,
            disable=~self.progress,
        ):

            self.calibrator(event)

            for tel_id in event.dl0.tel.keys():

                geom = self.event_source.subarray.tel[tel_id].camera.geometry
                dl1_tel = event.dl1.tel[tel_id]

                # Image cleaning
                image = dl1_tel.image  # Waiting for automatic gain selection
                mask = tailcuts_clean(geom, image, picture_thresh=10, boundary_thresh=5)
                cleaned = image.copy()
                cleaned[~mask] = 0

                # Image parametrisation
                params = hillas_parameters(geom, cleaned)

                # Save Ids, MC infos and Hillas informations
                self.writer.write(
                    geom.camera_name, [event.r0, event.simulation.shower, params]
                )

    def finish(self):
        self.log.info("End of job.")
        self.writer.close()


if __name__ == "__main__":
    tool = SimpleEventWriter()
    tool.run()
