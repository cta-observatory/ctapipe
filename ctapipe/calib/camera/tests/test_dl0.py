from ctapipe.calib.camera.dl0 import CameraDL0Reducer


def test_camera_dl0_reducer(example_event):
    telid = list(example_event.r1.tel)[0]
    reducer = CameraDL0Reducer()
    reducer.reduce(example_event)
    waveforms = example_event.dl0.tel[telid].waveform
    assert waveforms is not None


def test_check_r1_exists(example_event):
    telid = list(example_event.r1.tel)[0]
    reducer = CameraDL0Reducer()
    assert (reducer.check_r1_exists(example_event, telid) is True)
    example_event.r1.tel[telid].waveform = None
    assert (reducer.check_r1_exists(example_event, telid) is False)
