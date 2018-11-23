import numpy as np
from astropy import units as u
from ctapipe.instrument import CameraGeometry
from ctapipe.instrument.camera import (
    _find_neighbor_pixels,
    _get_min_pixel_seperation,
)
import pytest

cam_ids = CameraGeometry.get_known_camera_names()


def test_construct():
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    geom = CameraGeometry(cam_id=0, pix_id=np.arange(100),
                          pix_x=x * u.m, pix_y=y * u.m,
                          pix_area=x * u.m**2,
                          pix_type='rectangular',
                          pix_rotation="10d",
                          cam_rotation="12d")

    assert geom.cam_id == 0
    assert geom.pix_area is not None
    assert (geom.pix_rotation.deg - 10) < 1e-5
    assert (geom.cam_rotation.deg - 10) < 1e-5


def test_known_camera_names():
    cams = CameraGeometry.get_known_camera_names()
    assert len(cams) > 4
    assert 'FlashCam' in cams
    assert 'NectarCam' in cams

    for cam in cams:
        geom = CameraGeometry.from_name(cam)
        geom.info()


def test_make_rectangular_camera_geometry():
    geom = CameraGeometry.make_rectangular()
    assert geom.pix_x.shape == geom.pix_y.shape


def test_load_hess_camera():
    geom = CameraGeometry.from_name("LSTCam")
    assert len(geom.pix_x) == 1855


def test_guess_camera():
    px = np.linspace(-10, 10, 11328) * u.m
    py = np.linspace(-10, 10, 11328) * u.m
    geom = CameraGeometry.guess(px, py, 0 * u.m)
    assert geom.pix_type.startswith('rect')


def test_get_min_pixel_seperation():
    x, y = np.meshgrid(np.linspace(-5, 5, 5), np.linspace(-5, 5, 5))
    pixsep = _get_min_pixel_seperation(x.ravel(), y.ravel())
    assert pixsep == 2.5


def test_find_neighbor_pixels():
    x, y = np.meshgrid(np.linspace(-5, 5, 5), np.linspace(-5, 5, 5))
    neigh = _find_neighbor_pixels(x.ravel(), y.ravel(), rad=3.1)
    assert set(neigh[11]) == set([16, 6, 10, 12])


@pytest.mark.parametrize("cam_id", cam_ids)
def test_neighbor_pixels(cam_id):
    """
    test if each camera has a reasonable number of neighbor pixels (4 for
    rectangular, and 6 for hexagonal.  Other than edge pixels, the majority
    should have the same value
    """

    geom = CameraGeometry.from_name(cam_id)
    n_pix = len(geom.pix_id)
    n_neighbors = [len(x) for x in geom.neighbors]

    if geom.pix_type.startswith('hex'):
        assert n_neighbors.count(6) > 0.5 * n_pix
        assert n_neighbors.count(6) > n_neighbors.count(4)

    if geom.pix_type.startswith('rect'):
        assert n_neighbors.count(4) > 0.5 * n_pix
        assert n_neighbors.count(5) == 0
        assert n_neighbors.count(6) == 0

    assert n_neighbors.count(1) == 0  # no pixel should have a single neighbor


def test_to_and_from_table():
    geom = CameraGeometry.from_name("LSTCam")
    tab = geom.to_table()
    geom2 = geom.from_table(tab)

    assert geom.cam_id == geom2.cam_id
    assert (geom.pix_x == geom2.pix_x).all()
    assert (geom.pix_y == geom2.pix_y).all()
    assert (geom.pix_area == geom2.pix_area).all()
    assert geom.pix_type == geom2.pix_type


def test_write_read(tmpdir):

    filename = str(tmpdir.join('testcamera.fits.gz'))

    geom = CameraGeometry.from_name("LSTCam")
    geom.to_table().write(filename, overwrite=True)
    geom2 = geom.from_table(filename)

    assert geom.cam_id == geom2.cam_id
    assert (geom.pix_x == geom2.pix_x).all()
    assert (geom.pix_y == geom2.pix_y).all()
    assert (geom.pix_area == geom2.pix_area).all()
    assert geom.pix_type == geom2.pix_type


def test_precal_neighbors():
    geom = CameraGeometry(cam_id="TestCam",
                          pix_id=np.arange(3),
                          pix_x=np.arange(3) * u.deg,
                          pix_y=np.arange(3) * u.deg,
                          pix_area=np.ones(3) * u.deg**2,
                          neighbors=[
                              [1, ], [0, 2], [1, ]
                          ],
                          pix_type='rectangular',
                          pix_rotation="0deg",
                          cam_rotation="0deg")

    neigh = geom.neighbors
    assert len(neigh) == len(geom.pix_x)

    nmat = geom.neighbor_matrix
    assert nmat.shape == (len(geom.pix_x), len(geom.pix_x))


def test_slicing():
    geom = CameraGeometry.from_name("NectarCam")
    sliced1 = geom[100:200]

    assert len(sliced1.pix_x) == 100
    assert len(sliced1.pix_y) == 100
    assert len(sliced1.pix_area) == 100
    assert len(sliced1.pix_id) == 100

    sliced2 = geom[[5, 7, 8, 9, 10]]
    assert sliced2.pix_id[0] == 5
    assert sliced2.pix_id[1] == 7
    assert len(sliced2.pix_x) == 5


@pytest.mark.parametrize("cam_id", cam_ids)
def test_slicing_rotation(cam_id):
    cam = CameraGeometry.from_name(cam_id)
    cam.rotate('25d')

    sliced1 = cam[5:10]

    assert sliced1.pix_x[0] == cam.pix_x[5]


def test_border_pixels():
    from ctapipe.instrument.camera import CameraGeometry

    cam = CameraGeometry.from_name("LSTCam")

    assert np.sum(cam.get_border_pixel_mask(1)) == 168
    assert np.sum(cam.get_border_pixel_mask(2)) == 330

    cam = CameraGeometry.from_name("ASTRICam")
    assert np.sum(cam.get_border_pixel_mask(1)) == 212
    assert np.sum(cam.get_border_pixel_mask(2)) == 408

    assert cam.get_border_pixel_mask(1)[0]
    assert cam.get_border_pixel_mask(1)[2351]
    assert not cam.get_border_pixel_mask(1)[521]
