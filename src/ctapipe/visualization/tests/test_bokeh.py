"""Tests for the bokeh visualization"""
import numpy as np
import pytest

from ctapipe.coordinates import TelescopeFrame

bokeh = pytest.importorskip("bokeh")
bokeh_io = pytest.importorskip("bokeh.io")
output_file = bokeh_io.output_file
save = bokeh_io.save


def test_create_display_without_geometry(example_event, example_subarray):
    """Test we can create a display without giving the geometry to init"""
    from ctapipe.visualization.bokeh import CameraDisplay

    # test we can create it without geometry, and then set all the stuff
    display = CameraDisplay()

    tel_id = next(iter(example_event.r0.tel.keys()))
    display.geometry = example_subarray.tel[tel_id].camera.geometry
    display.image = example_event.dl1.tel[tel_id].image


def test_camera_display_creation(example_event, example_subarray):
    """Test we can create a display and check the resulting pixel coordinates"""
    from ctapipe.visualization.bokeh import CameraDisplay

    t = list(example_event.r0.tel.keys())[0]
    geom = example_subarray.tel[t].camera.geometry
    display = CameraDisplay(geom)

    assert np.allclose(np.mean(display.datasource.data["xs"], axis=1), geom.pix_x.value)
    assert np.allclose(np.mean(display.datasource.data["ys"], axis=1), geom.pix_y.value)


def test_camera_display_telescope_frame(example_event, example_subarray):
    """Test we can create a display in telescope frame"""
    from ctapipe.visualization.bokeh import CameraDisplay

    t = list(example_event.r0.tel.keys())[0]
    geom = example_subarray.tel[t].camera.geometry.transform_to(TelescopeFrame())
    display = CameraDisplay(geom)

    assert np.allclose(np.mean(display.datasource.data["xs"], axis=1), geom.pix_x.value)
    assert np.allclose(np.mean(display.datasource.data["ys"], axis=1), geom.pix_y.value)


def test_camera_image(example_event, example_subarray, tmp_path):
    """Test we set an image"""
    from ctapipe.visualization.bokeh import CameraDisplay

    t = list(example_event.r0.tel.keys())[0]
    geom = example_subarray.tel[t].camera.geometry
    image = np.ones(geom.n_pixels)

    display = CameraDisplay(geom, image)
    assert np.all(display.image == image)

    display.image = np.random.normal(size=geom.n_pixels)
    assert np.all(display.image == image)

    display.highlight_pixels(display.image > 0)

    output_path = tmp_path / "test.html"
    output_file(output_path)
    save(display.figure, filename=output_path)


def test_camera_enable_pixel_picker(example_event, example_subarray):
    """Test we can call enable_pixel_picker"""
    from ctapipe.visualization.bokeh import CameraDisplay

    t = list(example_event.r0.tel.keys())[0]
    geom = example_subarray.tel[t].camera.geometry
    n_pixels = geom.pix_x.value.size
    image = np.ones(n_pixels)
    c_display = CameraDisplay(geom, image)

    def callback(attr, new, old):
        print(attr, new, old)

    c_display.enable_pixel_picker(callback)


def test_matplotlib_cmaps(example_subarray):
    """Test using matplotlib colormap names works"""
    from ctapipe.visualization.bokeh import CameraDisplay

    geom = example_subarray.tel[1].camera.geometry
    image = np.ones(len(geom))
    display = CameraDisplay(geom, image)
    display.cmap = "viridis"
    display.cmap = "RdBu"


def test_cameras(camera_geometry, tmp_path):
    """Test for all known camera geometries"""
    from ctapipe.visualization.bokeh import CameraDisplay

    image = np.random.normal(size=len(camera_geometry))
    display = CameraDisplay(camera_geometry, image)

    output_path = tmp_path / "test.html"
    output_file(output_path)
    save(display.figure, filename=output_path)


def test_array_display_no_values(example_subarray, tmp_path):
    """Test plain array display"""
    from ctapipe.visualization.bokeh import ArrayDisplay

    display = ArrayDisplay(example_subarray)

    output_path = tmp_path / "test.html"
    output_file(output_path)
    save(display.figure, filename=output_path)


def test_array_display(example_subarray, tmp_path):
    """Test array display with values for each telescope"""
    from ctapipe.visualization.bokeh import ArrayDisplay

    display = ArrayDisplay(example_subarray, values=np.arange(len(example_subarray)))
    display.add_colorbar()

    output_path = tmp_path / "test.html"
    output_file(output_path)
    save(display.figure, filename=output_path)
