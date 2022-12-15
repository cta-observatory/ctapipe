from ctapipe.image import ImageProcessor
from ctapipe.io import EventSource


def test_processor(dl1_muon_file):
    from ctapipe.image.muon import MuonProcessor

    with EventSource(dl1_muon_file, focal_length_choice="EQUIVALENT") as source:
        image_processor = ImageProcessor(source.subarray)
        muon_processor = MuonProcessor(source.subarray)
        for event in source:
            image_processor(event)
            muon_processor(event)
