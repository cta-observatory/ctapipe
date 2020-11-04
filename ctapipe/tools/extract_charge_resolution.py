"""
Calculate the Charge Resolution from a sim_telarray simulation and store
within a HDF5 file.
"""

import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from traitlets import Dict, Int, List
from ctapipe.core.traits import Path

from ctapipe.analysis.camera.charge_resolution import ChargeResolutionCalculator
from ctapipe.calib import CameraCalibrator
from ctapipe.core import Provenance, Tool, traits
from ctapipe.image.extractor import ImageExtractor
from ctapipe.io.simteleventsource import SimTelEventSource


class ChargeResolutionGenerator(Tool):
    name = "ChargeResolutionGenerator"
    description = (
        "Calculate the Charge Resolution from a sim_telarray "
        "simulation and store within a HDF5 file."
    )

    telescopes = List(
        Int(),
        None,
        allow_none=True,
        help="Telescopes to include from the event file. Default = All telescopes",
    ).tag(config=True)
    output_path = Path(
        default_value="charge_resolution.h5",
        directory_ok=False,
        help="Path to store the output HDF5 file",
    ).tag(config=True)

    aliases = Dict(
        dict(
            f="SimTelEventSource.input_url",
            max_events="SimTelEventSource.max_events",
            T="SimTelEventSource.allowed_tels",
            O="ChargeResolutionGenerator.output_path",
        )
    )

    classes = List([SimTelEventSource] + traits.classes_with_traits(ImageExtractor))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eventsource = None
        self.calibrator = None
        self.calculator = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"

        self.eventsource = SimTelEventSource(parent=self)

        self.calibrator = CameraCalibrator(
            parent=self, subarray=self.eventsource.subarray
        )
        self.calculator = ChargeResolutionCalculator()

    def start(self):
        desc = "Extracting Charge Resolution"
        for event in tqdm(self.eventsource, desc=desc):
            self.calibrator(event)

            # Check events have true charge included
            if event.count == 0:
                try:
                    pe = list(event.simulation.tel.values())[0].true_image
                    if np.all(pe == 0):
                        raise KeyError
                except KeyError:
                    self.log.exception("Source does not contain true charge!")
                    raise

            for mc, dl1 in zip(event.simulation.tel.values(), event.dl1.tel.values()):
                true_charge = mc.true_image
                measured_charge = dl1.image
                pixels = np.arange(measured_charge.size)
                self.calculator.add(pixels, true_charge, measured_charge)

    def finish(self):
        df_p, df_c = self.calculator.finish()

        output_directory = os.path.dirname(self.output_path)
        if not os.path.exists(output_directory):
            self.log.info(f"Creating directory: {output_directory}")
            os.makedirs(output_directory)

        with pd.HDFStore(self.output_path, "w") as store:
            store["charge_resolution_pixel"] = df_p
            store["charge_resolution_camera"] = df_c

        self.log.info("Created charge resolution file: {}".format(self.output_path))
        Provenance().add_output_file(self.output_path)


def main():
    exe = ChargeResolutionGenerator()
    exe.run()


if __name__ == "__main__":
    main()
