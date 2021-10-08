from ctapipe.coordinates import TelescopeFrame
import numpy as np
from bokeh.io import save, output_file
import tempfile


def test_create_display_without_geometry(example_event, example_subarray):
    from ctapipe.visualization.bokeh import CameraDisplay

    # test we can create it without geometry, and then set all the stuff
    display = CameraDisplay()

    tel_id = next(iter(example_event.r0.tel.keys()))
    display.geometry = example_subarray.tel[tel_id].camera.geometry
    display.image = example_event.dl1.tel[tel_id].image


def test_camera_display_creation(example_event, example_subarray):
    from ctapipe.visualization.bokeh import CameraDisplay

    t = list(example_event.r0.tel.keys())[0]
    geom = example_subarray.tel[t].camera.geometry
    display = CameraDisplay(geom, autoshow=False)

    assert np.allclose(np.mean(display.datasource.data["xs"], axis=1), geom.pix_x.value)
    assert np.allclose(np.mean(display.datasource.data["ys"], axis=1), geom.pix_y.value)


def test_camera_display_telescope_frame(example_event, example_subarray):
    from ctapipe.visualization.bokeh import CameraDisplay

    t = list(example_event.r0.tel.keys())[0]
    geom = example_subarray.tel[t].camera.geometry.transform_to(TelescopeFrame())
    display = CameraDisplay(geom, autoshow=False)

    assert np.allclose(np.mean(display.datasource.data["xs"], axis=1), geom.pix_x.value)
    assert np.allclose(np.mean(display.datasource.data["ys"], axis=1), geom.pix_y.value)


def test_camera_image(example_event, example_subarray):
    from ctapipe.visualization.bokeh import CameraDisplay

    t = list(example_event.r0.tel.keys())[0]
    geom = example_subarray.tel[t].camera.geometry
    image = np.ones(geom.n_pixels)

    display = CameraDisplay(geom, image, autoshow=False)
    assert np.all(display.image == image)

    display.image = np.random.normal(size=geom.n_pixels)
    assert np.all(display.image == image)

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        output_file(f.name)
        save(display.figure, filename=f.name)


def test_camera_enable_pixel_picker(example_event, example_subarray):
    from ctapipe.visualization.bokeh import CameraDisplay

    t = list(example_event.r0.tel.keys())[0]
    geom = example_subarray.tel[t].camera.geometry
    n_pixels = geom.pix_x.value.size
    image = np.ones(n_pixels)
    c_display = CameraDisplay(geom, image, autoshow=False)

    def callback(attr, new, old):
        print(attr, new, old)

    c_display.enable_pixel_picker(callback)
