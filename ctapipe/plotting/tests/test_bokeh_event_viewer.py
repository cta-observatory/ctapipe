from ctapipe.plotting.bokeh_event_viewer import BokehEventViewer
from ctapipe.calib.camera.calibrator import CameraCalibrator
import pytest


def test_bokeh_event_viewer_creation():
    viewer = BokehEventViewer()
    viewer.create()


def test_event_setting(example_event):
    viewer = BokehEventViewer()
    viewer.create()
    viewer.event = example_event
    for cam in viewer.cameras:
        assert cam.event == example_event
    for wf in viewer.waveforms:
        assert wf.event == example_event


def test_enable_automatic_index_increment():
    viewer = BokehEventViewer()
    viewer.create()
    viewer.enable_automatic_index_increment()
    for cam in viewer.cameras:
        assert cam.automatic_index_increment


def test_change_time(example_event):
    viewer = BokehEventViewer()
    viewer.create()
    viewer.event = example_event

    t = 5
    viewer.change_time(t)
    for cam in viewer.cameras:
        assert cam.time == t
    for wf in viewer.waveforms:
        assert wf.active_time == t

    t = -11
    viewer.change_time(t)
    for cam in viewer.cameras:
        assert cam.time == 0
    for wf in viewer.waveforms:
        assert wf.active_time == 0

    tel = list(example_event.r0.tels_with_data)[0]
    n_samples = example_event.r0.tel[tel].waveform.shape[-1]
    t = 10000
    viewer.change_time(t)
    for cam in viewer.cameras:
        assert cam.time == n_samples - 1
    for wf in viewer.waveforms:
        assert wf.active_time == n_samples - 1


def test_on_waveform_click(example_event):
    viewer = BokehEventViewer()
    viewer.create()
    viewer.event = example_event

    t = 5
    viewer.waveforms[0]._on_waveform_click(t)
    for cam in viewer.cameras:
        assert cam.time == t
    for wf in viewer.waveforms:
        assert wf.active_time == t


def test_telid(example_event):
    viewer = BokehEventViewer()
    viewer.create()
    viewer.event = example_event

    tels = list(example_event.r0.tels_with_data)

    assert viewer.telid == tels[0]
    for cam in viewer.cameras:
        assert cam.telid == tels[0]
    for wf in viewer.waveforms:
        assert wf.telid == tels[0]

    viewer.telid = tels[1]
    assert viewer.telid == tels[1]
    for cam in viewer.cameras:
        assert cam.telid == tels[1]
    for wf in viewer.waveforms:
        assert wf.telid == tels[1]


def test_telid_incorrect(example_event):
    viewer = BokehEventViewer()
    viewer.create()
    viewer.event = example_event

    with pytest.raises(KeyError):
        viewer.telid = 148937242


def test_on_pixel_click(example_event):
    viewer = BokehEventViewer()
    viewer.create()
    viewer.event = example_event

    p1 = 5
    viewer.cameras[0]._on_pixel_click(p1)
    assert viewer.waveforms[viewer.cameras[0].active_index].pixel == p1


def test_channel(example_event):
    viewer = BokehEventViewer()
    viewer.create()
    viewer.event = example_event

    assert viewer.channel == 0
    for cam in viewer.cameras:
        assert cam.channel == 0
    for wf in viewer.waveforms:
        assert wf.channel == 0


def test_channel_incorrect(example_event):
    viewer = BokehEventViewer()
    viewer.create()
    viewer.event = example_event

    with pytest.raises(IndexError):
        viewer.channel = 148937242


def test_view_camera(example_event):
    viewer = BokehEventViewer()
    viewer.create()
    viewer.event = example_event

    calibrator = CameraCalibrator()
    calibrator(example_event)

    t = list(example_event.r0.tels_with_data)[0]

    cam = viewer.cameras[0]
    cam.view = 'r1'
    assert (cam.image == example_event.r1.tel[t].waveform[0, :, 0]).all()

    with pytest.raises(ValueError):
        cam.view = 'q'


def test_view_wf(example_event):
    viewer = BokehEventViewer()
    viewer.create()
    viewer.event = example_event

    calibrator = CameraCalibrator()
    calibrator(example_event)

    t = list(example_event.r0.tels_with_data)[0]

    wf = viewer.waveforms[0]
    wf.view = 'r1'
    assert (wf.waveform == example_event.r1.tel[t].waveform[0, 0, :]).all()

    with pytest.raises(ValueError):
        wf.view = 'q'
