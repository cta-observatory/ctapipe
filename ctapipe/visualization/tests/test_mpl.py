# skip these tests if matplotlib can't be imported
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from ctapipe.instrument import (CameraGeometry, SubarrayDescription,
                                TelescopeDescription)
from ctapipe.io.containers import HillasParametersContainer
from numpy import ones
from astropy import units as u


def test_camera_display_single():
    """ test CameraDisplay functionality """
    from ..mpl_camera import CameraDisplay

    geom = CameraGeometry.from_name("LSTCam")
    disp = CameraDisplay(geom)
    image = ones(len(geom.pix_x), dtype=float)
    disp.image = image
    disp.add_colorbar()
    disp.cmap = 'nipy_spectral'
    disp.set_limits_minmax(0, 10)
    disp.set_limits_percent(95)
    disp.enable_pixel_picker()
    disp.highlight_pixels([1, 2, 3, 4, 5])
    disp.norm = 'log'
    disp.norm = 'symlog'
    disp.cmap = 'rainbow'

    with pytest.raises(ValueError):
        disp.image = ones(10)

    with pytest.raises(ValueError):
        disp.add_colorbar()

    disp.add_ellipse(centroid=(0, 0), width=0.1, length=0.1, angle=0.1)
    disp.clear_overlays()


def test_camera_display_multiple():
    """ create a figure with 2 subplots, each with a CameraDisplay """
    from ..mpl_camera import CameraDisplay

    geom = CameraGeometry.from_name("LSTCam")
    fig, ax = plt.subplots(2, 1)

    d1 = CameraDisplay(geom, ax=ax[0])
    d2 = CameraDisplay(geom, ax=ax[1])

    image = ones(len(geom.pix_x), dtype=float)
    d1.image = image
    d2.image = image


def test_array_display():
    from ctapipe.visualization.mpl_array import ArrayDisplay

    # build a test subarray:
    tels = dict()
    tel_pos = dict()
    for ii, pos in enumerate([[0, 0, 0], [100, 0, 0], [-100, 0, 0]] * u.m):
        tels[ii + 1] = TelescopeDescription.from_name("MST", "NectarCam")
        tel_pos[ii + 1] = pos

    sub = SubarrayDescription(
        name="TestSubarray",
        tel_positions=tel_pos,
        tel_descriptions=tels
    )

    ad = ArrayDisplay(sub)
    ad.set_vector_rho_phi(1 * u.m, 90 * u.deg)

    # try setting a value
    vals = ones(sub.num_tels)
    ad.values = vals

    assert (vals == ad.values).all()

    # test using hillas params:
    hillas_dict = {
        1: HillasParametersContainer(length=1.0 * u.m, psi=90 * u.deg),
        2: HillasParametersContainer(length=200 * u.cm, psi="95deg"),
    }
    ad.set_vector_hillas(hillas_dict)
    ad.set_line_hillas(hillas_dict, range=300)
    ad.add_labels()
    ad.remove_labels()
