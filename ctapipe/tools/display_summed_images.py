"""
Select a group of similar telescopes (with same camera type), Loop over
events where telescopes in the group participate, sum the image from each
telescope, and display it.
"""

import numpy as np
from matplotlib import pyplot as plt

from ctapipe.calib import CameraCalibrator
from ctapipe.core import Tool
from ctapipe.core.traits import Unicode, Integer, Dict, List
from ctapipe.io import SimTelEventSource
from ctapipe.visualization import CameraDisplay


class ImageSumDisplayerTool(Tool):
    description = Unicode(__doc__)
    name = "ctapipe-display-imagesum"

    infile = Unicode(
        help='input simtelarray file',
        default="/Users/kosack/Data/CTA/Prod3/gamma.simtel.gz"
    ).tag(config=True)

    telgroup = Integer(
        help='telescope group number', default=1
    ).tag(config=True)

    max_events = Integer(
        help='stop after this many events if non-zero', default_value=0, min=0
    ).tag(config=True)

    output_suffix = Unicode(
        help='suffix (file extension) of output '
             'filenames to write images '
             'to (no writing is done if blank). '
             'Images will be named [EVENTID][suffix]',
        default_value=""
    ).tag(config=True)

    aliases = Dict({
        'infile': 'ImageSumDisplayerTool.infile',
        'telgroup': 'ImageSumDisplayerTool.telgroup',
        'max-events': 'ImageSumDisplayerTool.max_events',
        'output-suffix': 'ImageSumDisplayerTool.output_suffix'
    })

    classes = List([CameraCalibrator, SimTelEventSource])

    def setup(self):
        # load up the telescope types table (need to first open a file, a bit of
        # a hack until a proper insturment module exists) and select only the
        # telescopes with the same camera type
        # make sure gzip files are seekable

        self.reader = SimTelEventSource(
            input_url=self.infile, max_events=self.max_events, back_seekable=True
        )

        for event in self.reader:
            camtypes = event.inst.subarray.to_table().group_by('camera_type')
            event.inst.subarray.info(printer=self.log.info)
            break

        group = camtypes.groups[self.telgroup]
        self._selected_tels = list(group['tel_id'].data)
        self._base_tel = self._selected_tels[0]
        self.log.info(
            "Telescope group %d: %s", self.telgroup,
            str(event.inst.subarray.tel[self._selected_tels[0]])
        )
        self.log.info(f"SELECTED TELESCOPES:{self._selected_tels}")

        self.calibrator = CameraCalibrator(parent=self)

        self.reader.allowed_tels = self._selected_tels

    def start(self):
        geom = None
        imsum = None
        disp = None

        for event in self.reader:

            self.calibrator(event)

            if geom is None:
                geom = event.inst.subarray.tel[self._base_tel].camera
                imsum = np.zeros(shape=geom.pix_x.shape, dtype=np.float)
                disp = CameraDisplay(geom, title=geom.cam_id)
                disp.add_colorbar()
                disp.cmap = 'viridis'

            if len(event.dl0.tels_with_data) <= 2:
                continue

            imsum[:] = 0
            for telid in event.dl0.tels_with_data:
                imsum += event.dl1.tel[telid].image[0]

            self.log.info(
                "event={} ntels={} energy={}".format(
                    event.r0.event_id, len(event.dl0.tels_with_data),
                    event.mc.energy
                )
            )
            disp.image = imsum
            plt.pause(0.1)

            if self.output_suffix is not "":
                filename = "{:020d}{}".format(
                    event.r0.event_id, self.output_suffix
                )
                self.log.info(f"saving: '{filename}'")
                plt.savefig(filename)


def main():
    tool = ImageSumDisplayerTool()
    tool.run()
