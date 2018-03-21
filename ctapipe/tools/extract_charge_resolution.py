"""
Extract data necessary to calcualte charge resolution from raw data files.
"""

import os

import numpy as np
from tqdm import tqdm
from traitlets import Dict, List, Int, Unicode

from ctapipe.analysis.camera.chargeresolution import ChargeResolutionCalculator
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import HESSIOR1Calibrator
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from ctapipe.io.hessioeventsource import HESSIOEventSource


class ChargeResolutionGenerator(Tool):
    name = "ChargeResolutionGenerator"
    description = "Generate the a pickle file of ChargeResolutionFile for " \
                  "a MC file."

    telescopes = List(Int, None, allow_none=True,
                      help='Telescopes to include from the event file. '
                           'Default = All telescopes').tag(config=True)
    output_name = Unicode('charge_resolution',
                          help='Name of the output charge resolution hdf5 '
                               'file').tag(config=True)

    aliases = Dict(dict(f='HESSIOEventSource.input_url',
                        max_events='HESSIOEventSource.max_events',
                        extractor='ChargeExtractorFactory.product',
                        window_width='ChargeExtractorFactory.window_width',
                        t0='ChargeExtractorFactory.t0',
                        window_shift='ChargeExtractorFactory.window_shift',
                        sig_amp_cut_HG='ChargeExtractorFactory.sig_amp_cut_HG',
                        sig_amp_cut_LG='ChargeExtractorFactory.sig_amp_cut_LG',
                        lwt='ChargeExtractorFactory.lwt',
                        clip_amplitude='CameraDL1Calibrator.clip_amplitude',
                        radius='CameraDL1Calibrator.radius',
                        max_pe='ChargeResolutionCalculator.max_pe',
                        T='ChargeResolutionGenerator.telescopes',
                        O='ChargeResolutionGenerator.output_name',
                        ))
    classes = List([HESSIOEventSource,
                    ChargeExtractorFactory,
                    CameraDL1Calibrator,
                    ChargeResolutionCalculator
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eventsource = None
        self.r1 = None
        self.dl0 = None
        self.dl1 = None
        self.calculator = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.eventsource = HESSIOEventSource(**kwargs)

        extractor = ChargeExtractorFactory.produce(**kwargs)

        self.r1 = HESSIOR1Calibrator(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(extractor=extractor, **kwargs)

        self.calculator = ChargeResolutionCalculator(**kwargs)

    def start(self):
        desc = "Filling Charge Resolution"
        for event in tqdm(self.eventsource, desc=desc):
            tels = list(event.dl0.tels_with_data)

            # Check events have true charge included
            if event.count == 0:
                try:
                    if np.all(event.mc.tel[
                                  tels[0]].photo_electron_image == 0):
                        raise KeyError
                except KeyError:
                    self.log.exception('Source does not contain '
                                       'true charge!')
                    raise

            self.r1.calibrate(event)
            self.dl0.reduce(event)
            self.dl1.calibrate(event)

            if self.telescopes:
                tels = []
                for tel in self.telescopes:
                    if tel in event.dl0.tels_with_data:
                        tels.append(tel)

            for telid in tels:
                true_charge = event.mc.tel[telid].photo_electron_image
                measured_charge = event.dl1.tel[telid].image[0]
                self.calculator.add_charges(true_charge, measured_charge)

    def finish(self):
        input_url = self.eventsource.input_url
        input_directory = os.path.dirname(input_url)
        input_name = os.path.splitext(os.path.basename(input_url))[0]
        output_directory = os.path.join(input_directory, input_name)
        if not os.path.exists(output_directory):
            self.log.info("Creating directory: {}".format(output_directory))
            os.makedirs(output_directory)
        name = "{}.h5".format(self.output_name)
        ouput_path = os.path.join(output_directory, name)
        self.calculator.save(ouput_path)


def main():
    exe = ChargeResolutionGenerator()
    exe.run()


if __name__ == '__main__':
    main()
