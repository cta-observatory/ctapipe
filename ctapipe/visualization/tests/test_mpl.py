# skip these tests if matplotlib can't be imported
import pytest
plt = pytest.importorskip("matplotlib.pyplot")

from ctapipe.instrument import CameraGeometry
from ctapipe.io import get_array_layout
from numpy import ones

def test_camera_display_single():
    """ test CameraDisplay functionality """
    from ..mpl import CameraDisplay

    geom = CameraGeometry.from_name("LSTCam")
    disp = CameraDisplay(geom)
    image = ones(len(geom.pix_x), dtype=float)
    disp.image = image
    disp.add_colorbar()
    disp.cmap = 'spectral'
    disp.set_limits_minmax(0, 10)
    disp.set_limits_percent(95)


def test_camera_display_multiple():
    """ create a figure with 2 subplots, each with a CameraDisplay """
    from ..mpl import CameraDisplay
    
    geom = CameraGeometry.from_name("LSTCam")
    fig, ax = plt.subplots(2, 1)

    d1 = CameraDisplay(geom, ax=ax[0])
    d2 = CameraDisplay(geom, ax=ax[1])

    image = ones(len(geom.pix_x), dtype=float)
    d1.image = image
    d2.image = image


def test_array_display():
    from ..mpl import ArrayDisplay

    # load some test data
    layout = get_array_layout("hess")
    X = layout['POSX']
    Y = layout['POSY']
    A = layout['MIRAREA']
    A[:] = 132

    ad = ArrayDisplay(X, Y, A, title="HESS")
    ad.intensities = ones(len(X))
