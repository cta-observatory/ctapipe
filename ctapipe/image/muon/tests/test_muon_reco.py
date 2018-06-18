from ctapipe.image.muon import muon_reco_functions as muon
from ctapipe.calib import CameraCalibrator
from copy import deepcopy

def test_basic_muon_reco(test_event):
    event=deepcopy(test_event)

    calib = CameraCalibrator()
    calib.calibrate(event)

    muon_params = muon.analyze_muon_event(event)
    assert muon_params is not None