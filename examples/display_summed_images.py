"""
Select a group of similar telescopes (with same camera type), Loop over
events where telescopes in the group participate, sum the image from each
telescope, and display it
"""

import numpy as np
from astropy.table import Table
from ctapipe.core import Tool
from ctapipe.core.traits import *
from ctapipe.io.hessio import hessio_event_source
from ctapipe.visualization import CameraDisplay
from matplotlib import pyplot as plt
from ctapipe.calib import CameraCalibrator


class ImageSumDisplayerTool(Tool):
    description = Unicode(__doc__)
    name = "ctapipe-image-sum-display"

    infile = Unicode(help='input simtelarray file',
                     default="/Users/kosack/Data/CTA/Prod3/gamma.simtel.gz"
                     ).tag(config=True)
    telgroup = Integer(help='telescope group number', default=1).tag(
        config=True)

    max_events = Integer(help='stop after this many events if non-zero',
                         default_value=0, min=0).tag(config=True)

    output_suffix=Unicode(help='suffix (file extension) of output '
                               'filenames to write images '
                               'to (no writing is done if blank). '
                               'Images will be named [EVENTID][suffix]' ,
                          default_value="").tag(config=True)

    aliases = Dict({'infile': 'ImageSumDisplayerTool.infile',
                    'telgroup': 'ImageSumDisplayerTool.telgroup',
                    'max-events': 'ImageSumDisplayerTool.max_events',
                    'output-suffix': 'ImageSumDisplayerTool.output_suffix'})
    classes = List([CameraCalibrator,])

    def setup(self):
        # load up the telescope types table (need to first open a file, a bit of
        # a hack until a proper insturment module exists) and select only the
        # telescopes with the same camera type
        data = next(hessio_event_source(self.infile, max_events=1))
        data.inst.subarray.info(printer=self.log.info)
        camtypes = data.inst.subarray.to_table().group_by('camera_type')
        group = camtypes.groups[self.telgroup]
        self._selected_tels = group['tel_id'].data
        self._base_tel = self._selected_tels[0]
        self.log.info("Telescope group %d: %s",
                      self.telgroup,
                      str(data.inst.subarray.tel[self._selected_tels[0]]))
        self.log.info("SELECTED TELESCOPES:{}".format(self._selected_tels))
        self.calibrator = CameraCalibrator(self.config, self)

    def start(self):
        geom = None
        imsum = None
        disp = None

        for data in hessio_event_source(self.infile,
                                        allowed_tels=self._selected_tels,
                                        max_events=self.max_events):

            self.calibrator.calibrate(data)

            if geom is None:
                geom = data.inst.subarray.tel[self._base_tel].camera
                imsum = np.zeros(shape=geom.pix_x.shape, dtype=np.float)
                disp = CameraDisplay(geom, title=geom.cam_id)
                disp.add_colorbar()
                disp.cmap = 'viridis'

            if len(data.dl0.tels_with_data) <= 2:
                continue

            imsum[:] = 0
            for telid in data.dl0.tels_with_data:
                imsum += data.dl1.tel[telid].image[0]

            self.log.info("event={} ntels={} energy={}" \
                          .format(data.r0.event_id,
                                  len(data.dl0.tels_with_data),
                                  data.mc.energy))
            disp.image = imsum
            plt.pause(0.1)

            if self.output_suffix is not "":
                filename = "{:020d}{}".format(data.r0.event_id,
                                              self.output_suffix)
                self.log.info("saving: '{}'".format(filename))
                plt.savefig(filename)


if __name__ == '__main__':
    tool = ImageSumDisplayerTool()
    tool.run()
