"""
Select a group of similar telescopes (with same camera type), Loop over
events where telescopes in the group participate, sum the image from each
telescope, and display it
"""

import numpy as np
from astropy.table import Table
from ctapipe.core import Tool
from ctapipe.core.traits import (Unicode, Dict, Integer)
from ctapipe.instrument import CameraGeometry
from ctapipe.io.hessio import hessio_event_source
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument.camera import _guess_camera_type
from matplotlib import pyplot as plt


def get_camera_types(inst):
    """ return dict of group_id to list of telescopes in group,
    where a group is defined as similar telescopes"""
    data = dict(tel_id=[], tel_type=[], cam_type=[])
    for telid in inst.pixel_pos:
        x, y = inst.pixel_pos[telid]
        f = inst.optical_foclen[telid]
        tel_type, cam_type, pix_type, pix_rot, cam_rot = \
            _guess_camera_type(len(x), f)

        data['tel_type'].append(tel_type)
        data['tel_id'].append(int(telid))
        data['cam_type'].append(cam_type)

    return Table(data).group_by("cam_type")


class ImageSumDisplayerTool(Tool):
    description = Unicode(__doc__)
    name = "ctapipe-image-sum-display"

    infile = Unicode(help='input simtelarray file',
                     default="/Users/kosack/Data/CTA/Prod3/gamma.simtel.gz"
                     ).tag(config=True)
    telgroup = Integer(help='telescope group number', default=1).tag(
        config=True)

    aliases = Dict({'infile': 'ImageSumDisplayerTool.infile',
                    'telgroup': 'ImageSumDisplayerTool.telgroup'})

    def setup(self):
        # load up the telescope types table (need to first open a file, a bit of
        # a hack until a proper insturment module exists) and select only the
        # telescopes with the same camera type
        data = next(hessio_event_source(self.infile, max_events=1))
        camtypes = get_camera_types(data.inst)
        group = camtypes.groups[self.telgroup]
        self._selected_tels = group['tel_id'].data
        self._base_tel = self._selected_tels[0]
        self.log.info("Telescope group %d: %s with %s camera", self.telgroup,
                      group[0]['tel_type'], group[0]['cam_type'])
        self.log.info("SELECTED TELESCOPES:{}".format(self._selected_tels))

    def start(self):
        geom = None
        imsum = None
        disp = None

        for data in hessio_event_source(self.infile,
                                        allowed_tels=self._selected_tels,
                                        max_events=None):
            if geom is None:
                x, y = data.inst.pixel_pos[self._base_tel]
                flen = data.inst.optical_foclen[self._base_tel]
                geom = CameraGeometry.guess(x, y, flen)
                imsum = np.zeros(shape=x.shape, dtype=np.float)
                disp = CameraDisplay(geom, title=geom.cam_id)
                disp.add_colorbar()
                disp.cmap = 'viridis'

            if len(data.r0.tels_with_data) <= 2:
                continue

            imsum[:] = 0
            for telid in data.r0.tels_with_data:
                imsum += data.r0.tel[telid].adc_sums[0]

            self.log.info("event={} ntels={} energy={}" \
                          .format(data.r0.event_id,
                                  len(data.r0.tels_with_data),
                                  data.mc.energy))
            disp.image = imsum
            plt.pause(0.1)


if __name__ == '__main__':
    tool = ImageSumDisplayerTool()
    tool.run()
