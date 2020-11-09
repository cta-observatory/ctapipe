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
from ctapipe.containers import HillasParametersContainer
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

    # build a test subarray:
    tels = dict()
    tel_pos = dict()
    for ii, pos in enumerate([[0, 0, 0], [100, 0, 0], [-100, 0, 0]] * u.m):
        tels[ii + 1] = TelescopeDescription.from_name("MST", "NectarCam")
        tel_pos[ii + 1] = pos

    sub = SubarrayDescription(
        name="TestSubarray", tel_positions=tel_pos, tel_descriptions=tels
    )

    ad = ArrayDisplay(sub)
    ad.set_vector_rho_phi(1 * u.m, 90 * u.deg)

    # try setting a value
    vals = np.ones(sub.num_tels)
    ad.values = vals

    assert (vals == ad.values).all()

    # test using hillas params:
    hillas_dict = {
        1: HillasParametersContainer(length=100.0 * u.m, psi=90 * u.deg),
        2: HillasParametersContainer(length=20000 * u.cm, psi="95deg"),
    }

    grad = 2
    intercept = 1

    geom = CameraGeometry.from_name("LSTCam")
    rot_angle = 20 * u.deg
    hillas = HillasParametersContainer(x=0 * u.m, y=0 * u.m, psi=rot_angle)

    timing_rot20 = timing_parameters(
        geom,
        image=np.ones(geom.n_pixels),
        peak_time=intercept + grad * geom.pix_x.value,
        hillas_parameters=hillas,
        cleaning_mask=np.ones(geom.n_pixels, dtype=bool),
    )
    gradient_dict = {1: timing_rot20.slope.value, 2: timing_rot20.slope.value}
    ad.set_vector_hillas(
        hillas_dict=hillas_dict,
        length=500,
        time_gradient=gradient_dict,
        angle_offset=0 * u.deg,
    )

    ad.set_line_hillas(hillas_dict, range=300)
    ad.add_labels()
    ad.remove_labels()
