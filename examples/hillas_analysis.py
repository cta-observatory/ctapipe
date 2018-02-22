'''
This example shows how you should use ctapipe.core.tool.Tool
for your analysis as example for a simple Hillas based analysis.
'''
from ctapipe.core.tool import Tool
from ctapipe.core.traits import Unicode, Dict, List
from ctapipe.utils import get_dataset
from ctapipe.io.eventsourcefactory import EventSourceFactory
from ctapipe.calib import CameraCalibrator
from ctapipe.image.hillas import hillas_parameters
from ctapipe.image.cleaning import tailcuts_clean

import astropy.units as u

import json


class HillasAnalysis(Tool):
    # class members are options, that can be set via commandline or config file

    # inputfile, if None we will use the gamma test file
    inputfile = Unicode(None, allow_none=True).tag(config=True)

    # outputfile
    outputfile = Unicode('', allow_none=False).tag(config=True)

    classes = List([CameraCalibrator, EventSourceFactory])

    aliases = Dict({'outputfile': 'HillasAnalysis.outputfile'})

    def setup(self):
        print('inputfile', self.inputfile)
        print('outputfile', self.outputfile)
        self.event_source = EventSourceFactory.produce(
            tool=self, config=self.config,
            input_url=self.inputfile or get_dataset('gamma_test.simtel.gz')
        )
        self.calibrator = CameraCalibrator(
            tool=self,
            config=self.config,
            event_source=self.event_source
        )
        self.output_fh = open(self.outputfile, 'w')

    def start(self):
        for event in self.event_source:
            self.calibrator.calibrate(event)

            for telescope_id, dl1 in event.dl1.tel.items():
                camera = event.inst.subarray.tels[telescope_id].camera

                mask = tailcuts_clean(camera, dl1.image[0])
                hillas_params = hillas_parameters(camera[mask], dl1.image[0, mask])

                json.dump({
                    'event_id': event.count,
                    'telescope_id': int(telescope_id),
                    'camera_id': camera.cam_id,
                    'width_mm': hillas_params.width.to(u.mm).value,
                    'length_mm': hillas_params.length.to(u.mm).value,
                }, self.output_fh)
                self.output_fh.write('\n')

    def finish(self):
        self.output_fh.close()


def main():
    HillasAnalysis().run()


if __name__ == '__main__':
    main()
