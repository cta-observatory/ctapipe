from tqdm import tqdm
import numpy as np
from astropy.coordinates import SkyCoord
from ctapipe.containers import TelEventIndexContainer

from ctapipe.calib import CameraCalibrator
from ctapipe.core import Provenance
from ctapipe.core import Tool, ToolConfigurationError
from ctapipe.core import traits
from ctapipe.io import EventSource
from ctapipe.io import HDF5TableWriter
from ctapipe.image.cleaning import TailcutsImageCleaner
from ctapipe.coordinates import TelescopeFrame, CameraFrame
from ctapipe.containers import MuonParametersContainer
from ctapipe.instrument import CameraGeometry

from ctapipe.image.muon.muon_processor import MuonProcessor
from ctapipe.image.muon import (
    MuonRingFitter,
    MuonIntensityFitter,
    ring_containment,
    ring_completeness,
    intensity_ratio_inside_ring,
    mean_squared_error,
)


class MuonAnalysis(Tool):
    """
    Detect and extract muon ring parameters, and write the muon ring and
    intensity parameters to an output table.

    The resulting output can be read e.g. using for example
    `pandas.read_hdf(filename, 'dl1/event/telescope/parameters/muon')`
    """

    name = "ctapipe-reconstruct-muons"
    description = traits.Unicode(__doc__)

    output = traits.Path(directory_ok=False, help="HDF5 output file name").tag(
        config=True
    )

    overwrite = traits.Bool(
        default_value=False, help="If true, overwrite outputfile without asking"
    ).tag(config=True)

    classes = [
        MuonProcessor,
        CameraCalibrator,
        TailcutsImageCleaner,
        EventSource,
        MuonRingFitter,
        MuonIntensityFitter,
    ]

    aliases = {
        "i": "EventSource.input_url",
        "input": "EventSource.input_url",
        "o": "MuonAnalysis.output",
        "output": "MuonAnalysis.output",
        "max-events": "EventSource.max_events",
        "allowed-tels": "EventSource.allowed_tels",
    }

    flags = {
        "overwrite": ({"MuonAnalysis": {"overwrite": True}}, "overwrite output file")
    }

    def setup(self):
        if self.output is None:
            raise ToolConfigurationError("You need to provide an --output file")

        if self.output.exists() and not self.overwrite:
            raise ToolConfigurationError(
                "Outputfile {self.output} already exists, use `--overwrite` to overwrite"
            )
        
        self.source = EventSource(parent=self)
        subarray = self.source.subarray

        self.writer = HDF5TableWriter(
            self.output, "", add_prefix=True, parent=self, mode="w"
        )

        self.process_array_event = MuonProcessor(
            subarray=self.source.subarray,
            is_simulation=self.source.is_simulation,
            parent=self,
        )

    def start(self):
        for event in tqdm(self.source, desc="Processing events: "):
            self.process_array_event(event)

    def finish(self):
        Provenance().add_output_file(self.output, role="muon_efficiency_parameters")
        self.writer.close()


def main():
    tool = MuonAnalysis()
    tool.run()


if __name__ == "__main__":
    main()
