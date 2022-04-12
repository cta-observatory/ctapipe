"""
Tests for array display
"""

# skip these tests if matplotlib can't be imported
import pytest
from ctapipe.instrument import (
    CameraGeometry,
    SubarrayDescription,
    TelescopeDescription,
    PixelShape,
)
from ctapipe.containers import (
    CameraHillasParametersContainer,
    HillasParametersContainer,
)
import numpy as np
from astropy import units as u

plt = pytest.importorskip("matplotlib.pyplot")


def test_camera_display_single():
    """ test CameraDisplay functionality """
    from ..mpl_camera import CameraDisplay

    geom = CameraGeometry.from_name("LSTCam")
    disp = CameraDisplay(geom)
    image = np.random.normal(size=len(geom.pix_x))
    disp.image = image
    disp.add_colorbar()
    disp.cmap = "nipy_spectral"
    disp.set_limits_minmax(0, 10)
    disp.set_limits_percent(95)
    disp.enable_pixel_picker()
    disp.highlight_pixels([1, 2, 3, 4, 5])
    disp.norm = "log"
    disp.norm = "symlog"
    disp.cmap = "rainbow"

    with pytest.raises(ValueError):
        disp.image = np.ones(10)

    with pytest.raises(ValueError):
        disp.add_colorbar()

    disp.add_ellipse(centroid=(0, 0), width=0.1, length=0.1, angle=0.1)
    disp.clear_overlays()


def test_hillas_overlay():
    from ctapipe.visualization import CameraDisplay

    disp = CameraDisplay(CameraGeometry.from_name("LSTCam"))
    hillas = CameraHillasParametersContainer(
        x=0.1 * u.m, y=-0.1 * u.m, length=0.5 * u.m, width=0.2 * u.m, psi=90 * u.deg
    )

    disp.overlay_moments(hillas)


@pytest.mark.parametrize("pix_type", PixelShape.__members__.values())
def test_pixel_shapes(pix_type):
    """ test CameraDisplay functionality """
    from ..mpl_camera import CameraDisplay

    geom = CameraGeometry.from_name("LSTCam")
    geom.pix_type = pix_type

    disp = CameraDisplay(geom)
    image = np.random.normal(size=len(geom.pix_x))
    disp.image = image
    disp.add_colorbar()
    disp.highlight_pixels([1, 2, 3, 4, 5])
    disp.add_ellipse(centroid=(0, 0), width=0.1, length=0.1, angle=0.1)


def test_camera_display_multiple():
    """ create a figure with 2 subplots, each with a CameraDisplay """
    from ..mpl_camera import CameraDisplay

    geom = CameraGeometry.from_name("LSTCam")
    fig, ax = plt.subplots(2, 1)

    d1 = CameraDisplay(geom, ax=ax[0])
    d2 = CameraDisplay(geom, ax=ax[1])

    image = np.ones(len(geom.pix_x), dtype=float)
    d1.image = image
    d2.image = image


def test_array_display():
    """ check that we can do basic array display functionality """
    from ctapipe.visualization.mpl_array import ArrayDisplay
    from ctapipe.image import timing_parameters

    from ctapipe.containers import (
        ArrayEventContainer,
        DL1Container,
        DL1CameraContainer,
        ImageParametersContainer,
        CoreParametersContainer,
    )

    # build a test subarray:
    tels = dict()
    tel_pos = dict()
    for ii, pos in enumerate([[0, 0, 0], [100, 0, 0], [-100, 0, 0]] * u.m):
        tels[ii + 1] = TelescopeDescription.from_name("MST", "NectarCam")
        tel_pos[ii + 1] = pos

    sub = SubarrayDescription(
        name="TestSubarray", tel_positions=tel_pos, tel_descriptions=tels
    )

    # Create a fake event containing telescope-wise information about
    # the image directions projected on the ground
    event = ArrayEventContainer()
    event.dl1 = DL1Container()
    event.dl1.tel = {1: DL1CameraContainer(), 2: DL1CameraContainer()}
    event.dl1.tel[1].parameters = ImageParametersContainer()
    event.dl1.tel[2].parameters = ImageParametersContainer()
    event.dl1.tel[2].parameters.core = CoreParametersContainer()
    event.dl1.tel[1].parameters.core = CoreParametersContainer()
    event.dl1.tel[1].parameters.core.psi = u.Quantity(2.0, unit=u.deg)
    event.dl1.tel[2].parameters.core.psi = u.Quantity(1.0, unit=u.deg)

    ad = ArrayDisplay(subarray=sub)
    ad.set_vector_rho_phi(1 * u.m, 90 * u.deg)

    # try setting a value
    vals = np.ones(sub.num_tels)
    ad.values = vals

    assert (vals == ad.values).all()

    # test UV field ...

    # ...with colors by telescope type
    ad.set_vector_uv(np.array([1, 2, 3]) * u.m, np.array([1, 2, 3]) * u.m)
    # ...with scalar color
    ad.set_vector_uv(np.array([1, 2, 3]) * u.m, np.array([1, 2, 3]) * u.m, c=3)

    geom = CameraGeometry.from_name("LSTCam")
    rot_angle = 20 * u.deg
    hillas = CameraHillasParametersContainer(x=0 * u.m, y=0 * u.m, psi=rot_angle)

    # test using hillas params CameraFrame:
    hillas_dict = {
        1: CameraHillasParametersContainer(length=100.0 * u.m, psi=90 * u.deg),
        2: CameraHillasParametersContainer(length=20000 * u.cm, psi="95deg"),
    }

    grad = 2
    intercept = 1

    timing_rot20 = timing_parameters(
        geom,
        image=np.ones(geom.n_pixels),
        peak_time=intercept + grad * geom.pix_x.value,
        hillas_parameters=hillas,
        cleaning_mask=np.ones(geom.n_pixels, dtype=bool),
    )
    gradient_dict = {1: timing_rot20.slope.value, 2: timing_rot20.slope.value}
    core_dict = {
        tel_id: dl1.parameters.core.psi for tel_id, dl1 in event.dl1.tel.items()
    }
    ad.set_vector_hillas(
        hillas_dict=hillas_dict,
        core_dict=core_dict,
        length=500,
        time_gradient=gradient_dict,
        angle_offset=0 * u.deg,
    )
    ad.set_line_hillas(hillas_dict=hillas_dict, core_dict=core_dict, range=300)

    # test using hillas params for divergent pointing in telescopeframe:
    hillas_dict = {
        1: HillasParametersContainer(
            fov_lon=1.0 * u.deg, fov_lat=1.0 * u.deg, length=1.0 * u.deg, psi=90 * u.deg
        ),
        2: HillasParametersContainer(
            fov_lon=1.0 * u.deg, fov_lat=1.0 * u.deg, length=1.0 * u.deg, psi=95 * u.deg
        ),
    }
    ad.set_vector_hillas(
        hillas_dict=hillas_dict,
        core_dict=core_dict,
        length=500,
        time_gradient=gradient_dict,
        angle_offset=0 * u.deg,
    )
    ad.set_line_hillas(hillas_dict=hillas_dict, core_dict=core_dict, range=300)

    # test using hillas params for parallel pointing in telescopeframe:
    hillas_dict = {
        1: HillasParametersContainer(
            fov_lon=1.0 * u.deg, fov_lat=1.0 * u.deg, length=1.0 * u.deg, psi=90 * u.deg
        ),
        2: HillasParametersContainer(
            fov_lon=1.0 * u.deg, fov_lat=1.0 * u.deg, length=1.0 * u.deg, psi=95 * u.deg
        ),
    }
    ad.set_vector_hillas(
        hillas_dict=hillas_dict,
        core_dict=core_dict,
        length=500,
        time_gradient=gradient_dict,
        angle_offset=0 * u.deg,
    )

    # test negative time_gradients
    gradient_dict = {1: -0.03, 2: -0.02}
    ad.set_vector_hillas(
        hillas_dict=hillas_dict,
        core_dict=core_dict,
        length=500,
        time_gradient=gradient_dict,
        angle_offset=0 * u.deg,
    )
    # and very small
    gradient_dict = {1: 0.003, 2: 0.002}
    ad.set_vector_hillas(
        hillas_dict=hillas_dict,
        core_dict=core_dict,
        length=500,
        time_gradient=gradient_dict,
        angle_offset=0 * u.deg,
    )

    # Test background contour
    ad.background_contour(
        x=np.array([0, 1, 2]),
        y=np.array([0, 1, 2]),
        background=np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
    )

    ad.set_line_hillas(hillas_dict=hillas_dict, core_dict=core_dict, range=300)
    ad.add_labels()
    ad.remove_labels()


def test_picker():
    from ctapipe.visualization import CameraDisplay
    from matplotlib.backend_bases import MouseEvent, MouseButton

    geom = CameraGeometry.from_name("LSTCam")
    clicked_pixels = []

    class PickingCameraDisplay(CameraDisplay):
        def on_pixel_clicked(self, pix_id):
            print(f"YOU CLICKED PIXEL {pix_id}")
            clicked_pixels.append(pix_id)

    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])

    disp = PickingCameraDisplay(geom, ax=ax)
    disp.enable_pixel_picker()

    fig.canvas.draw()

    # emulate someone clicking the central pixel
    event = MouseEvent(
        "button_press_event", fig.canvas, x=500, y=500, button=MouseButton.LEFT
    )
    disp.pixels.pick(event)

    assert len(clicked_pixels) > 0
