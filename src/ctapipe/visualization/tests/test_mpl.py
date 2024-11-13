"""
Tests for array display
"""

import numpy as np

# skip these tests if matplotlib can't be imported
import pytest
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord

from ctapipe.calib.camera.calibrator import CameraCalibrator
from ctapipe.containers import (
    CameraHillasParametersContainer,
    HillasParametersContainer,
)
from ctapipe.coordinates.telescope_frame import TelescopeFrame
from ctapipe.instrument import PixelShape, SubarrayDescription

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture(scope="session")
def prod5_lst_cam(prod5_lst):
    return prod5_lst.camera.geometry


@pytest.mark.skipif(
    mpl.__version__ != "3.6.3",
    reason="See test below (test_camera_display_single)",
)
def test_norm_after_colorbar(prod5_lst_cam, tmp_path):
    """With matplotlib==3.6.3 we can not change the norm
    parameter of a CameraDisplay after we attached a colorbar."""
    from ..mpl_camera import CameraDisplay

    image = np.ones(prod5_lst_cam.pix_x.shape)

    fig, ax = plt.subplots()
    disp = CameraDisplay(prod5_lst_cam, ax=ax)
    disp.image = image
    disp.norm = "log"
    disp.add_colorbar()

    fig, ax = plt.subplots()
    disp = CameraDisplay(prod5_lst_cam, ax=ax)
    disp.image = image
    disp.add_colorbar()
    with pytest.raises(ValueError):
        disp.norm = "log"


@pytest.mark.skipif(
    mpl.__version__ == "3.6.3",
    reason=(
        "There is a problem in changing the norm after adding a colorbar. "
        + "This issue came up in #2207 and "
        + "should be fixed in a separate PR."
        + "See the test above (test_norm_after_colorbar)."
    ),
)
def test_camera_display_single(prod5_lst_cam, tmp_path):
    """test CameraDisplay functionality"""
    from ..mpl_camera import CameraDisplay

    fig, ax = plt.subplots()
    geom = prod5_lst_cam
    disp = CameraDisplay(geom, ax=ax)
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
    fig.savefig(tmp_path / "result.png")


def test_hillas_overlay_camera_frame(prod5_lst_cam, tmp_path):
    from ctapipe.visualization import CameraDisplay

    fig, ax = plt.subplots()
    disp = CameraDisplay(prod5_lst_cam, ax=ax)
    hillas = CameraHillasParametersContainer(
        x=0.1 * u.m, y=-0.1 * u.m, length=0.5 * u.m, width=0.2 * u.m, psi=90 * u.deg
    )

    disp.overlay_moments(hillas, color="w")
    fig.savefig(tmp_path / "result.png")


def test_hillas_overlay(prod5_lst_cam, tmp_path):
    from ctapipe.visualization import CameraDisplay

    fig, ax = plt.subplots()
    disp = CameraDisplay(prod5_lst_cam.transform_to(TelescopeFrame()), ax=ax)
    hillas = HillasParametersContainer(
        fov_lon=0.1 * u.deg,
        fov_lat=-0.1 * u.deg,
        length=0.5 * u.deg,
        width=0.2 * u.deg,
        psi=120 * u.deg,
    )

    disp.overlay_moments(hillas, color="w")
    fig.savefig(tmp_path / "result.png")


@pytest.mark.parametrize("pix_type", PixelShape.__members__.values())
def test_pixel_shapes(pix_type, prod5_lst_cam, tmp_path):
    """test CameraDisplay functionality"""
    from ..mpl_camera import CameraDisplay

    geom = prod5_lst_cam
    geom.pix_type = pix_type

    fig, ax = plt.subplots()
    disp = CameraDisplay(geom, ax=ax)
    image = np.random.normal(size=len(geom.pix_x))
    disp.image = image
    disp.add_colorbar()
    disp.highlight_pixels([1, 2, 3, 4, 5])
    disp.add_ellipse(centroid=(0, 0), width=0.1, length=0.1, angle=0.1)
    fig.savefig(tmp_path / "result.png")


def test_camera_display_multiple(prod5_lst_cam, tmp_path):
    """create a figure with 2 subplots, each with a CameraDisplay"""
    from ..mpl_camera import CameraDisplay

    geom = prod5_lst_cam
    fig, ax = plt.subplots(2, 1)

    d1 = CameraDisplay(geom, ax=ax[0])
    d2 = CameraDisplay(geom, ax=ax[1])

    image = np.ones(len(geom.pix_x), dtype=float)
    d1.image = image
    d2.image = image
    fig.savefig(tmp_path / "result.png")


def test_array_display(prod5_mst_nectarcam, reference_location):
    """check that we can do basic array display functionality"""
    from ctapipe.containers import (
        ArrayEventContainer,
        CoreParametersContainer,
        DL1CameraContainer,
        DL1Container,
        ImageParametersContainer,
    )
    from ctapipe.image import timing_parameters
    from ctapipe.visualization.mpl_array import ArrayDisplay

    # build a test subarray:
    tels = dict()
    tel_pos = dict()
    for ii, pos in enumerate([[0, 0, 0], [100, 0, 0], [-100, 0, 0]] * u.m):
        tels[ii + 1] = prod5_mst_nectarcam
        tel_pos[ii + 1] = pos

    sub = SubarrayDescription(
        name="TestSubarray",
        tel_positions=tel_pos,
        tel_descriptions=tels,
        reference_location=reference_location,
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
    vals = np.ones(sub.n_tels)
    ad.values = vals

    assert (vals == ad.values).all()

    # test UV field ...

    # ...with colors by telescope type
    ad.set_vector_uv(np.array([1, 2, 3]) * u.m, np.array([1, 2, 3]) * u.m)
    # ...with scalar color
    ad.set_vector_uv(np.array([1, 2, 3]) * u.m, np.array([1, 2, 3]) * u.m, c=3)

    geom = prod5_mst_nectarcam.camera.geometry
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


def test_picker(prod5_lst_cam):
    from matplotlib.backend_bases import MouseButton, MouseEvent

    from ctapipe.visualization import CameraDisplay

    geom = prod5_lst_cam
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


def test_overlay_coord(tmp_path, subarray_and_event_gamma_off_axis_500_gev):
    from ctapipe.visualization import CameraDisplay

    subarray, event = subarray_and_event_gamma_off_axis_500_gev

    calib = CameraCalibrator(subarray)
    calib(event)

    pointing = AltAz(
        alt=event.pointing.array_altitude,
        az=event.pointing.array_azimuth,
    )

    # add pointing here, so the transform to CameraFrame / TelescopeFrame works
    true_coord = SkyCoord(
        alt=event.simulation.shower.alt,
        az=event.simulation.shower.az,
        telescope_pointing=pointing,
        frame=AltAz(),
    )

    geometry = subarray.tel[1].camera.geometry
    image = event.dl1.tel[1].image

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

    display = CameraDisplay(geometry, ax=ax1)
    display.image = image
    display.overlay_coordinate(
        true_coord, color="xkcd:yellow", markeredgecolor="k", mew=1, ms=10
    )

    geometry_tel_frame = geometry.transform_to(TelescopeFrame())
    display = CameraDisplay(geometry_tel_frame, ax=ax2)
    display.image = image
    display.overlay_coordinate(
        true_coord, color="xkcd:yellow", markeredgecolor="k", mew=1, ms=10
    )

    fig.savefig(tmp_path / "coord_overlay.png", dpi=300)


@pytest.mark.parametrize("layout", (None, "constrained"))
def test_array_display_axes(tmp_path, subarray_prod5_paranal, layout):
    """Test passing axes to peek"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), layout=layout)

    subarray_prod5_paranal.peek(ax=ax1)
    subarray_prod5_paranal.peek(ax=ax2)

    fig.savefig(tmp_path / "double_peek.png")
