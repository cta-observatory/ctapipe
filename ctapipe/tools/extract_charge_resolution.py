"""
Calculate the Charge Resolution from a sim_telarray simulation and store
within a HDF5 file.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from traitlets import Dict, List, Unicode
from ctapipe.analysis.camera.charge_resolution import \
    ChargeResolutionCalculator
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import HESSIOR1Calibrator
from ctapipe.core import Tool, Provenance
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from ctapipe.io.simteleventsource import SimTelEventSource


class ChargeResolutionGenerator(Tool):
    name = "ChargeResolutionGenerator"
    description = ("Calculate the Charge Resolution from a sim_telarray "
                   "simulation and store within a HDF5 file.")

    output_path = Unicode(
        'charge_resolution.h5',
        help='Path to store the output HDF5 file'
    ).tag(config=True)

    aliases = Dict(dict(
        f='SimTelEventSource.input_url',
        max_events='SimTelEventSource.max_events',
        T='SimTelEventSource.allowed_tels',
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
        o='ChargeResolutionGenerator.output_path',
    ))
    classes = List([
        SimTelEventSource,
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

        self.eventsource = SimTelEventSource(**kwargs)

        extractor = ChargeExtractorFactory.produce(**kwargs)

        self.r1 = HESSIOR1Calibrator(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(extractor=extractor, **kwargs)

        self.calculator = ChargeResolutionCalculator()

    def start(self):
        desc = "Extracting Charge Resolution"
        for event in tqdm(self.eventsource, desc=desc):
            self.r1.calibrate(event)
            self.dl0.reduce(event)
            self.dl1.calibrate(event)

            # Check events have true charge included
            if event.count == 0:
                try:
                    pe = list(event.mc.tel.values())[0].photo_electron_image
                    if np.all(pe == 0):
                        raise KeyError
                except KeyError:
                    self.log.exception(
                        'Source does not contain true charge!'
                    )
                    raise

            for mc, dl1 in zip(event.mc.tel.values(), event.dl1.tel.values()):
                true_charge = mc.photo_electron_image
                measured_charge = dl1.image[0]
                pixels = np.arange(measured_charge.size)
                self.calculator.add(pixels, true_charge, measured_charge)

    def finish(self):
        df_p, df_c = self.calculator.finish()

        output_directory = os.path.dirname(self.output_path)
        if not os.path.exists(output_directory):
            self.log.info("Creating directory: {}".format(output_directory))
            os.makedirs(output_directory)

        with pd.HDFStore(self.output_path, 'w') as store:
            store['charge_resolution_pixel'] = df_p
            store['charge_resolution_camera'] = df_c

        self.log.info("Created charge resolution file: {}"
                      .format(self.output_path))
        Provenance().add_output_file(self.output_path)


def main():
    exe = ChargeResolutionGenerator()
    exe.run()


if __name__ == '__main__':
    main()
