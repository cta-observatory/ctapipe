""" Tests for CameraGeometry """
from copy import deepcopy

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord

from ctapipe.instrument import CameraGeometry, PixelShape
from ctapipe.instrument.warnings import FromNameWarning


def test_construct():
    """Check we can make a CameraGeometry from scratch"""
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    geom = CameraGeometry(
        name="Unknown",
        pix_id=np.arange(100),
        pix_x=x * u.m,
        pix_y=y * u.m,
        pix_area=x * u.m**2,
        pix_type=PixelShape.SQUARE,
        pix_rotation="10d",
        cam_rotation="12d",
    )

    assert geom.name == "Unknown"
    assert geom.pix_area is not None
    assert (geom.pix_rotation.deg - 10) < 1e-5
    assert (geom.cam_rotation.deg - 10) < 1e-5

    with pytest.raises(TypeError):
        geom = CameraGeometry(
            name="Unknown",
            pix_id=np.arange(100),
            pix_x=x * u.m,
            pix_y=y * u.m,
            pix_area=x * u.m**2,
            pix_type="foo",
            pix_rotation="10d",
            cam_rotation="12d",
        )

    # test from string:
    geom = CameraGeometry(
        name="Unknown",
        pix_id=np.arange(100),
        pix_x=x * u.m,
        pix_y=y * u.m,
        pix_area=x * u.m**2,
        pix_type="rectangular",
        pix_rotation="10d",
        cam_rotation="12d",
    )


def test_make_rectangular_camera_geometry():
    """Check that we can construct a dummy camera with square geometry"""
    geom = CameraGeometry.make_rectangular()
    assert geom.pix_x.shape == geom.pix_y.shape


def test_load_lst_camera(prod5_lst):
    """test that a specific camera has the expected attributes"""
    assert len(prod5_lst.camera.geometry.pix_x) == 1855
    assert prod5_lst.camera.geometry.pix_type == PixelShape.HEXAGON


def test_position_to_pix_index(prod5_lst):
    """test that we can lookup a pixel from a coordinate"""
    geometry = prod5_lst.camera.geometry

    x, y = (0.80 * u.m, 0.79 * u.m)

    pix_id = geometry.position_to_pix_index(x, y)

    assert pix_id == 1575

    pix_ids = geometry.position_to_pix_index([0.8, 0.8] * u.m, [0.79, 0.79] * u.m)
    np.testing.assert_array_equal(pix_ids, [1575, 1575])

    assert len(geometry.position_to_pix_index([] * u.m, [] * u.m)) == 0
    assert geometry.position_to_pix_index(5 * u.m, 5 * u.m) == np.iinfo(int).min


def test_find_neighbor_pixels():
    """test basic neighbor functionality"""
    n_pixels_grid = 5
    x, y = u.Quantity(
        np.meshgrid(
            np.linspace(-5, 5, n_pixels_grid), np.linspace(-5, 5, n_pixels_grid)
        ),
        u.cm,
    )
    x = x.ravel()
    y = y.ravel()
    n_pixels = len(x)

    geom = CameraGeometry(
        "test",
        pix_id=np.arange(n_pixels),
        pix_area=u.Quantity(np.full(n_pixels, 4), u.cm**2),
        pix_x=x.ravel(),
        pix_y=y.ravel(),
        pix_type="rectangular",
    )

    neigh = geom.neighbors
    assert set(neigh[11]) == {16, 6, 10, 12}


def test_neighbor_pixels(camera_geometry):
    """
    test if each camera has a reasonable number of neighbor pixels (4 for
    rectangular, and 6 for hexagonal.  Other than edge pixels, the majority
    should have the same value
    """
    geom = camera_geometry
    n_pix = len(geom.pix_id)
    n_neighbors = [len(x) for x in geom.neighbors]

    if geom.pix_type == PixelShape.HEXAGON:
        assert n_neighbors.count(6) > 0.5 * n_pix
        assert n_neighbors.count(6) > n_neighbors.count(4)

    if geom.pix_type == PixelShape.SQUARE:
        assert n_neighbors.count(4) > 0.5 * n_pix
        assert n_neighbors.count(5) == 0
        assert n_neighbors.count(6) == 0

    # whipple has inhomogenious pixels that mess with pixel neighborhood
    # calculation
    if not geom.name.startswith("Whipple"):
        assert np.all(geom.neighbor_matrix == geom.neighbor_matrix.T)
        assert n_neighbors.count(1) == 0  # no pixel should have a single neighbor


def test_calc_pixel_neighbors_square():
    x, y = np.meshgrid(np.arange(20), np.arange(20))

    cam = CameraGeometry(
        name="test",
        pix_id=np.arange(400),
        pix_type="rectangular",
        pix_x=u.Quantity(x.ravel(), u.cm),
        pix_y=u.Quantity(y.ravel(), u.cm),
        pix_area=u.Quantity(np.ones(400), u.cm**2),
    )

    assert set(cam.neighbors[0]) == {1, 20}
    assert set(cam.neighbors[21]) == {1, 20, 22, 41}


def test_calc_pixel_neighbors_square_diagonal():
    """
    check that neighbors for square-pixel cameras are what we expect,
    namely that the diagonals are included if requested.
    """
    x, y = np.meshgrid(np.arange(20), np.arange(20))

    cam = CameraGeometry(
        name="test",
        pix_id=np.arange(400),
        pix_type="rectangular",
        pix_x=u.Quantity(x.ravel(), u.cm),
        pix_y=u.Quantity(y.ravel(), u.cm),
        pix_area=u.Quantity(np.ones(400), u.cm**2),
    )

    cam._neighbors = cam.calc_pixel_neighbors(diagonal=True)
    assert set(cam.neighbors[21]) == {0, 1, 2, 20, 22, 40, 41, 42}


def test_to_and_from_table(prod5_lst):
    """Check converting to and from an astropy Table"""
    prod5_lst_cam = prod5_lst.camera.geometry
    tab = prod5_lst_cam.to_table()
    prod5_lst_cam2 = prod5_lst_cam.from_table(tab)

    assert prod5_lst_cam.name == prod5_lst_cam2.name
    assert (prod5_lst_cam.pix_x == prod5_lst_cam2.pix_x).all()
    assert (prod5_lst_cam.pix_y == prod5_lst_cam2.pix_y).all()
    assert (prod5_lst_cam.pix_area == prod5_lst_cam2.pix_area).all()
    assert prod5_lst_cam.pix_type == prod5_lst_cam2.pix_type


def test_write_read(tmpdir, prod5_lst):
    """Check that serialization to disk doesn't lose info"""
    filename = str(tmpdir.join("testcamera.fits.gz"))
    prod5_lst_cam = prod5_lst.camera.geometry

    prod5_lst_cam.to_table().write(filename, overwrite=True)
    prod5_lst_cam2 = prod5_lst_cam.from_table(filename)

    assert prod5_lst_cam.name == prod5_lst_cam2.name
    assert (prod5_lst_cam.pix_x == prod5_lst_cam2.pix_x).all()
    assert (prod5_lst_cam.pix_y == prod5_lst_cam2.pix_y).all()
    assert (prod5_lst_cam.pix_area == prod5_lst_cam2.pix_area).all()
    assert prod5_lst_cam.pix_type == prod5_lst_cam2.pix_type


def test_precal_neighbors():
    """
    test that pre-calculated neighbor lists don't get
    overwritten by automatic ones
    """
    geom = CameraGeometry(
        name="TestCam",
        pix_id=np.arange(3),
        pix_x=np.arange(3) * u.deg,
        pix_y=np.arange(3) * u.deg,
        pix_area=np.ones(3) * u.deg**2,
        neighbors=[[1], [0, 2], [1]],
        pix_type="rectangular",
        pix_rotation="0deg",
        cam_rotation="0deg",
    )

    neigh = geom.neighbors
    assert len(neigh) == len(geom.pix_x)

    nmat = geom.neighbor_matrix
    assert nmat.shape == (len(geom.pix_x), len(geom.pix_x))
    assert np.all(nmat.T == nmat)


def test_slicing(prod5_mst_nectarcam):
    """Check that we can slice a camera into a smaller one"""
    prod5_nectarcam = prod5_mst_nectarcam.camera.geometry
    sliced1 = prod5_nectarcam[100:200]

    assert len(sliced1.pix_x) == 100
    assert len(sliced1.pix_y) == 100
    assert len(sliced1.pix_area) == 100
    assert len(sliced1.pix_id) == 100

    sliced2 = prod5_nectarcam[[5, 7, 8, 9, 10]]
    assert sliced2.pix_id[0] == 5
    assert sliced2.pix_id[1] == 7
    assert len(sliced2.pix_x) == 5


def test_slicing_rotation(camera_geometry):
    """Check that we can rotate and slice"""
    camera_geometry.rotate("25d")

    sliced1 = camera_geometry[5:10]

    assert sliced1.pix_x[0] == camera_geometry.pix_x[5]


def test_rectangle_patch_neighbors():
    """ " test that a simple rectangular camera has the expected neighbors"""
    pix_x = np.array([-1.1, 0.1, 0.9, -1, 0, 1, -0.9, -0.1, 1.1]) * u.m
    pix_y = np.array([1.1, 1, 0.9, -0.1, 0, 0.1, -0.9, -1, -1.1]) * u.m
    pix_area = np.full(len(pix_x), 0.01) * u.m**2
    cam = CameraGeometry(
        name="testcam",
        pix_id=np.arange(pix_x.size),
        pix_x=pix_x,
        pix_y=pix_y,
        pix_area=pix_area,
        pix_type="rectangular",
    )

    assert np.all(cam.neighbor_matrix.T == cam.neighbor_matrix)
    assert cam.neighbor_matrix.sum(axis=0).max() == 4
    assert cam.neighbor_matrix.sum(axis=0).min() == 2


def test_border_pixels(prod5_lst, prod3_astri):
    """check we can find border pixels"""
    prod5_lst_cam = prod5_lst.camera.geometry

    assert np.sum(prod5_lst_cam.get_border_pixel_mask(1)) == 168
    assert np.sum(prod5_lst_cam.get_border_pixel_mask(2)) == 330

    prod3_astri_cam = prod3_astri.camera.geometry
    assert np.sum(prod3_astri_cam.get_border_pixel_mask(1)) == 212
    assert np.sum(prod3_astri_cam.get_border_pixel_mask(2)) == 408

    assert prod3_astri_cam.get_border_pixel_mask(1)[0]
    assert prod3_astri_cam.get_border_pixel_mask(1)[2351]
    assert not prod3_astri_cam.get_border_pixel_mask(1)[521]


def test_equals(prod5_lst, prod5_mst_nectarcam):
    """check we can use the == operator"""
    cam1 = prod5_lst.camera.geometry
    cam2 = deepcopy(prod5_lst.camera.geometry)
    cam3 = prod5_mst_nectarcam.camera.geometry

    assert cam1 is not cam2
    assert cam1 == cam2
    assert cam1 != cam3


def test_hashing(prod5_lst, prod5_mst_nectarcam):
    """ " check that hashes are correctly computed"""
    cam1 = prod5_lst.camera.geometry
    cam2 = deepcopy(prod5_lst.camera.geometry)
    cam3 = prod5_mst_nectarcam.camera.geometry

    assert len(set([cam1, cam2, cam3])) == 2


def test_camera_from_name(camera_geometry):
    """check we can construct all cameras from name"""
    with pytest.warns(FromNameWarning):
        camera = CameraGeometry.from_name(camera_geometry.name)
    assert str(camera) == camera_geometry.name


def test_camera_coordinate_transform(camera_geometry):
    """test conversion of the coordinates stored in a camera frame"""
    from ctapipe.coordinates import (
        CameraFrame,
        EngineeringCameraFrame,
        NominalFrame,
        TelescopeFrame,
    )

    geom = camera_geometry
    trans_geom = geom.transform_to(EngineeringCameraFrame())

    unit = geom.pix_x.unit
    assert np.allclose(geom.pix_x.to_value(unit), -trans_geom.pix_y.to_value(unit))
    assert np.allclose(geom.pix_y.to_value(unit), -trans_geom.pix_x.to_value(unit))

    # also test converting into a spherical frame:
    focal_length = 1.2 * u.m
    geom.frame = CameraFrame(focal_length=focal_length)

    pointing_position = SkyCoord(alt=70 * u.deg, az=0 * u.deg, frame=AltAz())
    telescope_frame = TelescopeFrame(telescope_pointing=pointing_position)
    geom_tel_frame = geom.transform_to(telescope_frame)

    x = geom_tel_frame.pix_x.to_value(u.deg)
    assert len(x) == len(geom.pix_x)

    # nominal frame with large offset, regression test for #2028
    origin = pointing_position.directional_offset_by(0 * u.deg, 5 * u.deg)
    nominal_frame = NominalFrame(origin=origin)

    geom_nominal = geom_tel_frame.transform_to(nominal_frame)
    # test that pixel sizes are still the same, i.e. calculation is taking translation into account
    assert u.allclose(geom_nominal.pix_area, geom_tel_frame.pix_area, rtol=0.01)

    # and test going backward from spherical to cartesian:

    geom_cam = geom_tel_frame.transform_to(CameraFrame(focal_length=focal_length))
    assert np.allclose(geom_cam.pix_x.to_value(unit), geom.pix_x.to_value(unit))


def test_guess_width():
    x = u.Quantity([0, 1, 2], u.cm)
    y = u.Quantity([0, 0, 0], u.cm)

    assert u.isclose(CameraGeometry.guess_pixel_width(x, y), 1 * u.cm)


def test_pixel_width():
    geom = CameraGeometry(
        "test",
        pix_id=[1],
        pix_area=[2] * u.cm**2,
        pix_x=[0] * u.m,
        pix_y=[0] * u.m,
        pix_type="hex",
    )

    assert np.isclose(geom.pixel_width.to_value(u.cm), [2 * np.sqrt(1 / np.sqrt(3))])

    geom = CameraGeometry(
        "test",
        pix_id=[1],
        pix_area=[2] * u.cm**2,
        pix_x=[0] * u.m,
        pix_y=[0] * u.m,
        pix_type="rect",
    )

    assert np.isclose(geom.pixel_width.to_value(u.cm), [np.sqrt(2)])


def test_guess_radius(prod5_lst, prod5_sst):
    prod5_lst_cam = prod5_lst.camera.geometry
    assert u.isclose(prod5_lst_cam.guess_radius(), 1.1 * u.m, rtol=0.05)

    prod5_chec = prod5_sst.camera.geometry
    assert u.isclose(prod5_chec.guess_radius(), 0.16 * u.m, rtol=0.05)


def test_single_pixel(prod5_lst):
    """Regression test for #2316"""
    single_pixel = prod5_lst.camera.geometry[[0]]

    assert single_pixel.neighbor_matrix.shape == (1, 1)
    assert single_pixel.neighbor_matrix[0, 0]


def test_empty(prod5_lst):
    geometry = prod5_lst.camera.geometry
    mask = np.zeros(len(geometry), dtype=bool)
    empty = geometry[mask]

    assert empty.neighbor_matrix.shape == (0, 0)
