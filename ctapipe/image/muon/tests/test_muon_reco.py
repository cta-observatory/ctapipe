from ctapipe.calib import CameraCalibrator
from ctapipe.image.muon import muon_reco_functions as muon


def test_basic_muon_reco(example_event):
    """
    Really simplistic test: just run the analyze_muon_event code, to make
    sure it doesn't crash. The input event is so far not a muon, so no output
    is generated.

    Parameters
    ----------
    test_event - a sample event (fixture)

    """

    calib = CameraCalibrator()
    calib.calibrate(example_event)

    muon_params = muon.analyze_muon_event(example_event)
    assert muon_params is not None
