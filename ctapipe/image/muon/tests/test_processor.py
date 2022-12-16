import numpy as np

from ctapipe.image import ImageProcessor
from ctapipe.io import EventSource


def test_processor(dl1_muon_file):
    from ctapipe.image.muon import MuonProcessor

    with EventSource(dl1_muon_file, focal_length_choice="EQUIVALENT") as source:
        image_processor = ImageProcessor(source.subarray)
        muon_processor = MuonProcessor(source.subarray)

        efficiencies = []
        for event in source:
            image_processor(event)
            muon_processor(event)
            for tel_id in event.dl1.tel:
                efficiencies.append(
                    event.muon.tel[tel_id].efficiency.optical_efficiency
                )

        assert len(efficiencies) > 0  # Assert there were events analyzed
        assert np.any(
            np.isfinite(efficiencies)
        )  # Assert at least some were completely analyzed
        assert np.all(
            np.logical_or(np.isfinite(efficiencies), np.isnan(efficiencies))
        )  # Assert all were at least provided defaults
