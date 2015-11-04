from ..mpl import CameraDisplay, ArrayDisplay
from ctapipe import io
from numpy import ones
from matplotlib import pyplot as plt


def test_camera_display_single():
    """ test CameraDisplay functionality """
    geom = io.CameraGeometry.from_name("HESS", 1)
    disp = CameraDisplay(geom)
    image = ones(len(geom.pix_x), dtype=float)
    disp.image = image
    disp.add_colorbar()
    disp.cmap = 'spectral'
    disp.set_limits_minmax(0, 10)
    disp.set_limits_percent(95)


def test_camera_display_multiple():
    """ create a figure with 2 subplots, each with a CameraDisplay """
    geom = io.CameraGeometry.from_name("HESS", 1)
    fig, ax = plt.subplots(2, 1)

    d1 = CameraDisplay(geom, axes=ax[0])
    d2 = CameraDisplay(geom, axes=ax[1])

    image = ones(len(geom.pix_x), dtype=float)
    d1.image = image
    d2.image = image


def test_array_display():

    # load some test data
    layout = io.get_array_layout("hess")
    X = layout['POSX']
    Y = layout['POSY']
    A = layout['MIRAREA']
    A[:] = 132

    ad = ArrayDisplay(X, Y, A, title="HESS")
    ad.intensities = ones(len(X))
