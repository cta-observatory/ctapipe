import numpy as np
import pytest


def test_camera_display_create():
    from ctapipe.visualization.bokeh import CameraDisplay
    CameraDisplay()


def test_camera_geom(example_event):
    from ctapipe.visualization.bokeh import CameraDisplay

    t = list(example_event.r0.tels_with_data)[0]
    geom = example_event.inst.subarray.tel[t].camera
    c_display = CameraDisplay(geom)

    assert (c_display.cdsource.data['x'] == geom.pix_x.value).all()
    assert (c_display.cdsource.data['y'] == geom.pix_y.value).all()

    t = list(example_event.r0.tels_with_data)[1]
    geom = example_event.inst.subarray.tel[t].camera
    c_display.geom = geom
    assert (c_display.cdsource.data['x'] == geom.pix_x.value).all()
    assert (c_display.cdsource.data['y'] == geom.pix_y.value).all()


def test_camera_image(example_event):
    from ctapipe.visualization.bokeh import CameraDisplay, intensity_to_hex

    t = list(example_event.r0.tels_with_data)[0]
    geom = example_event.inst.subarray.tel[t].camera
    n_pixels = geom.pix_x.value.size
    image = np.ones(n_pixels)
    colors = intensity_to_hex(image)

    with pytest.raises(ValueError):
        CameraDisplay(None, image)

    c_display = CameraDisplay(geom, image)
    assert (c_display.cdsource.data['image'] == colors).all()
    assert c_display.image_min == 0
    assert c_display.image_max == 2

    image[5] = 5
    colors = intensity_to_hex(image)
    c_display.image = image
    assert (c_display.cdsource.data['image'] == colors).all()
    assert c_display.image_min == image.min()
    assert c_display.image_max == image.max()


def test_camera_enable_pixel_picker(example_event):
    from ctapipe.visualization.bokeh import CameraDisplay
    t = list(example_event.r0.tels_with_data)[0]
    geom = example_event.inst.subarray.tel[t].camera
    n_pixels = geom.pix_x.value.size
    image = np.ones(n_pixels)
    c_display = CameraDisplay(geom, image)

    c_display.enable_pixel_picker(2)
    assert len(c_display.active_pixels) == 2

    c_display.enable_pixel_picker(3)
    assert len(c_display.active_pixels) == 3


def test_fast_camera_display_create(example_event):
    from ctapipe.visualization.bokeh import FastCameraDisplay
    t = list(example_event.r0.tels_with_data)[0]
    geom = example_event.inst.subarray.tel[t].camera

    x = geom.pix_x.value
    y = geom.pix_y.value
    area = geom.pix_area.value
    size = np.sqrt(area)

    FastCameraDisplay(x, y, size)


def test_fast_camera_image(example_event):
    from ctapipe.visualization.bokeh import FastCameraDisplay, intensity_to_hex

    t = list(example_event.r0.tels_with_data)[0]
    geom = example_event.inst.subarray.tel[t].camera

    x = geom.pix_x.value
    y = geom.pix_y.value
    area = geom.pix_area.value
    size = np.sqrt(area)

    c_display = FastCameraDisplay(x, y, size)

    image = np.ones(x.size)
    colors = intensity_to_hex(image)
    c_display.image = colors

    assert (c_display.cdsource.data['image'] == colors).all()


def test_waveform_display_create():
    from ctapipe.visualization.bokeh import WaveformDisplay

    WaveformDisplay()


def test_waveform_values():
    from ctapipe.visualization.bokeh import WaveformDisplay

    wf = np.ones(30)
    w_display = WaveformDisplay(wf)

    assert (w_display.cdsource.data['samples'] == wf).all()
    assert (w_display.cdsource.data['t'] == np.arange(wf.size)).all()

    wf[5] = 5
    w_display.waveform = wf

    assert (w_display.cdsource.data['samples'] == wf).all()
    assert (w_display.cdsource.data['t'] == np.arange(wf.size)).all()


def test_span():
    from ctapipe.visualization.bokeh import WaveformDisplay

    wf = np.ones(30)
    w_display = WaveformDisplay(wf)
    w_display.enable_time_picker()
    w_display.active_time = 4
    assert w_display.span.location == 4

    w_display.active_time = -3
    assert w_display.active_time == 0
    assert w_display.span.location == 0

    w_display.active_time = wf.size + 10
    assert w_display.active_time == wf.size - 1
    assert w_display.span.location == wf.size - 1
