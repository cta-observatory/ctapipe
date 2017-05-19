"""
Extract data necessary to calcualte charge resolution from raw data files.
"""

import os

import numpy as np
from astropy.utils.console import ProgressBarOrSpinner
from traitlets import Dict, List, Int, Unicode

from ctapipe.analysis.camera.chargeresolution import ChargeResolutionCalculator
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from ctapipe.io.eventfilereader import HessioFileReader


class ChargeResolutionGenerator(Tool):
    name = "ChargeResolutionGenerator"
    description = "Generate the a pickle file of ChargeResolutionFile for " \
                  "a MC file."

    telescopes = List(Int, None, allow_none=True,
                      help='Telescopes to include from the event file. '
                           'Default = All telescopes').tag(config=True)
    output_name = Unicode('charge_resolution',
                          help='Name of the output charge resolution pickle '
                               'file').tag(config=True)

    aliases = Dict(dict(f='HessioFileReader.input_path',
                        max_events='HessioFileReader.max_events',
                        extractor='ChargeExtractorFactory.extractor',
                        window_width='ChargeExtractorFactory.window_width',
                        window_start='ChargeExtractorFactory.window_start',
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
    classes = List([HessioFileReader,
                    ChargeExtractorFactory,
                    CameraDL1Calibrator,
                    ChargeResolutionCalculator
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_reader = None
        self.r1 = None
        self.dl0 = None
        self.dl1 = None
        self.calculator = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.file_reader = HessioFileReader(**kwargs)

        extractor_factory = ChargeExtractorFactory(**kwargs)
        extractor_class = extractor_factory.get_class()
        extractor = extractor_class(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin=self.file_reader.origin,
                                               **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(extractor=extractor, **kwargs)

        self.calculator = ChargeResolutionCalculator(**kwargs)

    def start(self):
        desc = "Filling Charge Resolution"
        with ProgressBarOrSpinner(None, message=desc) as pbar:
            source = self.file_reader.read()
            for event in source:
                pbar.update(event.count)
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
        directory = self.file_reader.output_directory
        name = "{}.pickle".format(self.output_name)
        ouput_path = os.path.join(directory, name)
        self.calculator.save(ouput_path)


def main():
    exe = ChargeResolutionGenerator()
    exe.run()


if __name__ == '__main__':
    main()
