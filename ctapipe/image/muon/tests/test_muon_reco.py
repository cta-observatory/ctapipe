from copy import deepcopy

from ctapipe.calib import CameraCalibrator
from ctapipe.image.muon import muon_reco_functions as muon


def test_basic_muon_reco(test_event):
    """
    Really simplistic test: just run the analyze_muon_event code, to make
    sure it doesn't crash. The input event is so far not a muon, so no output
    is generated.

    Parameters
    ----------
    test_event - a sample event (fixture)

    """
    event = deepcopy(test_event)

    calib = CameraCalibrator()
    calib.calibrate(event)

    muon_params = muon.analyze_muon_event(event)
    assert muon_params is not None
