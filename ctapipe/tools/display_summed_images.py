"""
Select a group of similar telescopes (with same camera type), Loop over
events where telescopes in the group participate, sum the image from each
telescope, and display it.
"""

import numpy as np
from matplotlib import pyplot as plt

from ctapipe.calib import CameraCalibrator
from ctapipe.core import Tool
from ctapipe.core.traits import Unicode, Integer, Dict, List, Path
from ctapipe.io import SimTelEventSource
from ctapipe.visualization import CameraDisplay
from ctapipe.utils import get_dataset_path


class ImageSumDisplayerTool(Tool):
    description = Unicode(__doc__)
    name = "ctapipe-display-imagesum"

    infile = Path(
        help="input simtelarray file",
        default_value=get_dataset_path("gamma_test_large.simtel.gz"),
        exists=True,
        directory_ok=False,
    ).tag(config=True)

    telgroup = Integer(help="telescope group number", default_value=1).tag(config=True)

    max_events = Integer(
        help="stop after this many events if non-zero", default_value=0, min=0
    ).tag(config=True)

    output_suffix = Unicode(
        help="suffix (file extension) of output "
        "filenames to write images "
        "to (no writing is done if blank). "
        "Images will be named [EVENTID][suffix]",
        default_value="",
    ).tag(config=True)

    aliases = Dict(
        {
            "infile": "ImageSumDisplayerTool.infile",
            "telgroup": "ImageSumDisplayerTool.telgroup",
            "max-events": "ImageSumDisplayerTool.max_events",
            "output-suffix": "ImageSumDisplayerTool.output_suffix",
        }
    )

    classes = List([CameraCalibrator, SimTelEventSource])

    def setup(self):
        # load up the telescope types table (need to first open a file, a bit of
        # a hack until a proper instrument module exists) and select only the
        # telescopes with the same camera type
        # make sure gzip files are seekable

        self.reader = SimTelEventSource(
            input_url=self.infile, max_events=self.max_events, back_seekable=True
        )

        camtypes = self.reader.subarray.to_table().group_by("camera_type")
        self.reader.subarray.info(printer=self.log.info)

        group = camtypes.groups[self.telgroup]
        self._selected_tels = list(group["tel_id"].data)
        self._base_tel = self._selected_tels[0]
        self.log.info(
            "Telescope group %d: %s",
            self.telgroup,
            str(self.reader.subarray.tel[self._selected_tels[0]]),
        )
        self.log.info(f"SELECTED TELESCOPES:{self._selected_tels}")

        self.calibrator = CameraCalibrator(parent=self, subarray=self.reader.subarray)

        self.reader.allowed_tels = self._selected_tels

    def start(self):
        geom = None
        imsum = None
        disp = None

        for event in self.reader:

            self.calibrator(event)

            if geom is None:
                geom = self.reader.subarray.tel[self._base_tel].camera.geometry
                imsum = np.zeros(shape=geom.pix_x.shape, dtype=np.float)
                disp = CameraDisplay(geom, title=geom.camera_name)
                disp.add_colorbar()
                disp.cmap = "viridis"

            if len(event.dl0.tel.keys()) <= 2:
                continue

            imsum[:] = 0
            for telid in event.dl0.tel.keys():
                imsum += event.dl1.tel[telid].image

            self.log.info(
                "event={} ntels={} energy={}".format(
                    event.index.event_id,
                    len(event.dl0.tel.keys()),
                    event.simulation.shower.energy,
                )
            )
            disp.image = imsum
            plt.pause(0.1)

            if self.output_suffix != "":
                filename = "{:020d}{}".format(event.index.event_id, self.output_suffix)
                self.log.info(f"saving: '{filename}'")
                plt.savefig(filename)


def main():
    tool = ImageSumDisplayerTool()
    tool.run()
