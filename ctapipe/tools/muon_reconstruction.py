from tqdm import tqdm

from ctapipe.calib import CameraCalibrator
from ctapipe.core import Tool
from ctapipe.core import traits
from ctapipe.io import EventSource
from ctapipe.io import DL1Writer
from ctapipe.image.cleaning import TailcutsImageCleaner

from ctapipe.image.muon.muon_processor import MuonProcessor
from ctapipe.image.muon import MuonRingFitter, MuonIntensityFitter


class MuonAnalysis(Tool):
    """
    Detect and extract muon ring parameters, and write the muon ring and
    intensity parameters to an output table.

    The resulting output can be read e.g. using for example
    `pandas.read_hdf(filename, 'dl1/event/telescope/parameters/muon')`
    """

    name = "ctapipe-reconstruct-muons"
    description = traits.Unicode(__doc__)

    classes = [
        MuonProcessor,
        CameraCalibrator,
        TailcutsImageCleaner,
        EventSource,
        MuonRingFitter,
        MuonIntensityFitter,
        DL1Writer,
    ]

    aliases = {
        ("i", "input"): "EventSource.input_url",
        ("o", "output"): "DL1Writer.output_path",
        "max-events": "EventSource.max_events",
        "allowed-tels": "EventSource.allowed_tels",
    }

    flags = {"overwrite": ({"DL1Writer": {"overwrite": True}}, "overwrite output file")}

    def setup(self):
        self.source = EventSource(parent=self)

        self.write_dl1 = DL1Writer(event_source=self.source, parent=self, is_muon=True)

        self.process_array_event = MuonProcessor(
            subarray=self.source.subarray, parent=self
        )

    def start(self):
        for event in tqdm(self.source, desc="Processing events: "):
            self.process_array_event(event)
            self.write_dl1(event)

    def finish(self):
        self.write_dl1.finish()


def main():
    tool = MuonAnalysis()
    tool.run()


if __name__ == "__main__":
    main()
