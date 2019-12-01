from ctapipe.image.muon.muon_reco_functions import analyze_muon_event


def test_analyze_muon_event(calibrated_event):
    """
    Really simplistic test: just run the analyze_muon_event code, to make
    sure it doesn't crash. The input event is so far not a muon, so no output
    is generated.

    Parameters
    ----------
    calibrated_event - a sample event (fixture)
    """
    muon_params = analyze_muon_event(calibrated_event)
