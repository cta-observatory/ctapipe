import numpy as np
from astropy import units as u
from ctapipe.instrument import CameraGeometry
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


def test_position_to_pix_index():
    geom = CameraGeometry.from_name("LSTCam")
    x, y = 0.80 * u.m, 0.79 * u.m,
    pix_id = geom.position_to_pix_index(x, y)
    assert pix_id == 1790


def test_find_neighbor_pixels():
    n_pixels = 5
    x, y = u.Quantity(np.meshgrid(
        np.linspace(-5, 5, n_pixels),
        np.linspace(-5, 5, n_pixels)
    ), u.cm)

    geom = CameraGeometry(
        'test',
        pix_id=np.arange(n_pixels),
        pix_area=u.Quantity(4, u.cm**2),
        pix_x=x.ravel(),
        pix_y=y.ravel(),
        pix_type='rectangular',
    )

    neigh = geom.neighbors
    assert set(neigh[11]) == {16, 6, 10, 12}


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

    # whipple has inhomogenious pixels that mess with pixel neighborhood
    # calculation
    if cam_id != 'Whipple490':
        assert np.all(geom.neighbor_matrix == geom.neighbor_matrix.T)
        assert n_neighbors.count(1) == 0  # no pixel should have a single neighbor


def test_calc_pixel_neighbors_square():

    x, y = np.meshgrid(np.arange(20), np.arange(20))

    cam = CameraGeometry(
        cam_id='test',
        pix_id=np.arange(400),
        pix_type='rectangular',
        pix_x=u.Quantity(x.ravel(), u.cm),
        pix_y=u.Quantity(y.ravel(), u.cm),
        pix_area=u.Quantity(np.ones(400), u.cm**2),
    )

    assert set(cam.neighbors[0]) == {1, 20}
    assert set(cam.neighbors[21]) == {1, 20, 22, 41}


def test_calc_pixel_neighbors_square_diagonal():
    x, y = np.meshgrid(np.arange(20), np.arange(20))

    cam = CameraGeometry(
        cam_id='test',
        pix_id=np.arange(400),
        pix_type='rectangular',
        pix_x=u.Quantity(x.ravel(), u.cm),
        pix_y=u.Quantity(y.ravel(), u.cm),
        pix_area=u.Quantity(np.ones(400), u.cm**2),
    )

    cam._neighbors = cam.calc_pixel_neighbors(diagonal=True)
    assert set(cam.neighbors[21]) == {0, 1, 2, 20, 22, 40, 41, 42}


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
    assert np.all(nmat.T == nmat)


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


def test_rectangle_patch_neighbors():
    pix_x = np.array([
        -1.1, 0.1, 0.9,
        -1, 0, 1,
        -0.9, -0.1, 1.1
    ]) * u.m
    pix_y = np.array([
        1.1, 1, 0.9,
        -0.1, 0, 0.1,
        -0.9, -1, -1.1
    ]) * u.m
    cam = CameraGeometry(
        cam_id='testcam',
        pix_id=np.arange(pix_x.size),
        pix_x=pix_x,
        pix_y=pix_y,
        pix_area=None,
        pix_type='rectangular',
    )

    assert np.all(cam.neighbor_matrix.T == cam.neighbor_matrix)
    assert cam.neighbor_matrix.sum(axis=0).max() == 4
    assert cam.neighbor_matrix.sum(axis=0).min() == 2


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


def test_equals():
    cam1 = CameraGeometry.from_name("LSTCam")
    cam2 = CameraGeometry.from_name("LSTCam")
    cam3 = CameraGeometry.from_name("ASTRICam")

    assert cam1 is not cam2
    assert cam1 == cam2
    assert cam1 != cam3


def test_hashing():
    cam1 = CameraGeometry.from_name("LSTCam")
    cam2 = CameraGeometry.from_name("LSTCam")
    cam3 = CameraGeometry.from_name("ASTRICam")

    assert len(set([cam1, cam2, cam3])) == 2
