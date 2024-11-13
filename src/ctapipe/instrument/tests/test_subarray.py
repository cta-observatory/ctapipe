""" Tests for SubarrayDescriptions """
from copy import deepcopy

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.earth import EarthLocation

from ctapipe.coordinates import TelescopeFrame
from ctapipe.instrument import (
    CameraDescription,
    OpticsDescription,
    SubarrayDescription,
    TelescopeDescription,
)

LOCATION = EarthLocation(lon=-17 * u.deg, lat=28 * u.deg, height=2200 * u.m)


def create_subarray(tel_type, n_tels=10):
    """generate a simple subarray for testing purposes"""
    rng = np.random.default_rng(0)

    pos = {}
    tel = {}

    for tel_id in range(1, n_tels + 1):
        tel[tel_id] = tel_type
        pos[tel_id] = rng.uniform(-100, 100, size=3) * u.m

    return SubarrayDescription(
        "test array",
        tel_positions=pos,
        tel_descriptions=tel,
        reference_location=LOCATION,
    )


def test_subarray_description(prod5_mst_nectarcam):
    """Test SubarrayDescription functionality"""
    n_tels = 10
    sub = create_subarray(prod5_mst_nectarcam, n_tels)

    assert len(sub.telescope_types) == 1

    assert str(sub) == "test array"
    assert sub.n_tels == n_tels
    assert len(sub.tel_ids) == n_tels
    assert sub.tel_ids[0] == 1
    assert sub.tel[1].camera is not None
    assert 0 not in sub.tel  # check that there is no tel 0 (1 is first above)
    assert len(sub.camera_types) == 1  # only 1 camera type
    assert isinstance(sub.camera_types[0], CameraDescription)
    assert isinstance(sub.telescope_types[0], TelescopeDescription)
    assert isinstance(sub.optics_types[0], OpticsDescription)
    assert len(sub.telescope_types) == 1  # only have one type in this array
    assert len(sub.optics_types) == 1  # only have one type in this array
    assert len(sub.camera_types) == 1  # only have one type in this array
    assert u.isclose(sub.optics_types[0].equivalent_focal_length, 16.0 * u.m)
    assert u.isclose(sub.optics_types[0].effective_focal_length, 16.445 * u.m)
    assert isinstance(sub.tel_coords, SkyCoord)
    assert len(sub.tel_coords) == n_tels

    assert sub.tel_coords.reference_location == LOCATION

    subsub = sub.select_subarray([2, 3, 4, 6], name="newsub")
    assert subsub.n_tels == 4
    assert set(subsub.tels.keys()) == {2, 3, 4, 6}
    assert subsub.tel_indices[6] == 3
    assert subsub.tel_ids[3] == 6

    assert len(sub.to_table(kind="optics")) == 1
    assert sub.telescope_types[0] == sub.tel[1]


def test_subarray_peek(prod5_mst_nectarcam):
    pytest.importorskip("matplotlib")
    sub = create_subarray(prod5_mst_nectarcam, 10)
    sub.peek()


def test_to_table(example_subarray):
    """Check that we can generate astropy Tables from the SubarrayDescription"""
    sub = example_subarray
    assert len(sub.to_table(kind="subarray")) == sub.n_tels
    assert len(sub.to_table(kind="optics")) == len(sub.optics_types)


def test_tel_indexing(example_subarray):
    """Check that we can convert between telescope_id and telescope_index"""
    sub = example_subarray

    assert sub.tel_indices[1] == 0  # first tel_id is in slot 0
    for tel_id in sub.tel_ids:
        assert sub.tel_index_array[tel_id] == sub.tel_indices[tel_id]

    assert sub.tel_ids_to_indices(1) == 0
    assert np.all(sub.tel_ids_to_indices([1, 2, 3]) == np.array([0, 1, 2]))


def test_tel_ids_to_mask(prod5_lst, reference_location):
    subarray = SubarrayDescription(
        "someone_counted_in_binary",
        tel_positions={1: [0, 0, 0] * u.m, 10: [50, 0, 0] * u.m},
        tel_descriptions={1: prod5_lst, 10: prod5_lst},
        reference_location=reference_location,
    )

    assert np.all(subarray.tel_ids_to_mask([]) == [False, False])
    assert np.all(subarray.tel_ids_to_mask([1]) == [True, False])
    assert np.all(subarray.tel_ids_to_mask([10]) == [False, True])
    assert np.all(subarray.tel_ids_to_mask([1, 10]) == [True, True])


def test_get_tel_ids_for_type(example_subarray):
    """
    check that we can get a list of telescope ids by a telescope type, which can
    be passed by string or `TelescopeDescription` instance
    """
    sub = example_subarray
    types = sub.telescope_types

    for teltype in types:
        assert len(sub.get_tel_ids_for_type(teltype)) > 0
        assert len(sub.get_tel_ids_for_type(str(teltype))) > 0

    with pytest.raises(ValueError):
        sub.get_tel_ids_for_type("NON_EXISTENT_TYPE")


def test_hdf(example_subarray, tmp_path):
    import tables

    path = tmp_path / "subarray.h5"

    example_subarray.to_hdf(path)
    read = SubarrayDescription.from_hdf(path, focal_length_choice="EQUIVALENT")

    assert example_subarray == read

    # test we can write the read subarray
    read.to_hdf(path, overwrite=True)

    # test we have a frame attached to the geometry with correction information
    for tel_id, tel in read.tel.items():
        assert (
            tel.camera.geometry.frame.focal_length == tel.optics.equivalent_focal_length
        )
        # test if transforming works
        tel.camera.geometry.transform_to(TelescopeFrame())

    # Test we can also write and read to an already opened h5file
    with tables.open_file(path, "w") as h5file:
        example_subarray.to_hdf(h5file)

    with tables.open_file(path, "r") as h5file:
        read = SubarrayDescription.from_hdf(h5file, focal_length_choice="EQUIVALENT")
        assert read == example_subarray


def test_hdf_same_camera(tmp_path, prod5_lst, prod5_mst_flashcam, reference_location):
    """Test writing / reading subarray to hdf5 with a subarray that has two
    different telescopes with the same camera
    """
    frankenstein_lst = TelescopeDescription(
        name="LST",
        optics=prod5_lst.optics,
        camera=prod5_mst_flashcam.camera,
    )
    tel = {
        1: prod5_lst,
        2: frankenstein_lst,
    }
    pos = {1: [0, 0, 0] * u.m, 2: [50, 0, 0] * u.m}

    array = SubarrayDescription(
        "test array",
        tel_positions=pos,
        tel_descriptions=tel,
        reference_location=reference_location,
    )

    path = tmp_path / "subarray.h5"
    array.to_hdf(path)
    read = SubarrayDescription.from_hdf(path, focal_length_choice="EQUIVALENT")
    assert array == read


def test_hdf_duplicate_string_repr(tmp_path, prod5_lst, reference_location):
    """Test writing and reading of a subarray with two telescopes that
    are different but have the same name.
    """
    # test with a subarray that has two different telescopes with the same
    # camera
    tel1 = prod5_lst

    # second telescope is almost the same and as the same str repr
    tel2 = deepcopy(tel1)
    # e.g. one mirror fell off
    tel2.optics.n_mirror_tiles = tel1.optics.n_mirror_tiles - 1

    array = SubarrayDescription(
        "test array",
        tel_positions={1: [0, 0, 0] * u.m, 2: [50, 0, 0] * u.m},
        tel_descriptions={1: tel1, 2: tel2},
        reference_location=reference_location,
    )

    # defensive checks to make sure we are actually testing this
    assert len(array.telescope_types) == 2
    assert str(tel1) == str(tel2)
    assert tel1 != tel2

    path = tmp_path / "subarray.h5"
    array.to_hdf(path)
    read = SubarrayDescription.from_hdf(path, focal_length_choice="EQUIVALENT")
    assert array == read
    assert read.tel[1].optics.n_mirror_tiles == read.tel[2].optics.n_mirror_tiles + 1


def test_get_tel_ids(example_subarray, prod3_astri):
    """Test for SubarrayDescription.get_tel_ids"""
    subarray = example_subarray

    telescopes = [1, 2, "MST_MST_FlashCam", prod3_astri]
    tel_ids = subarray.get_tel_ids(telescopes)

    true_tel_ids = (
        subarray.get_tel_ids_for_type("MST_MST_FlashCam")
        + subarray.get_tel_ids_for_type(prod3_astri)
        + (1, 2)
    )

    assert sorted(tel_ids) == sorted(true_tel_ids)

    # test invalid telescope type
    with pytest.raises(Exception):
        subarray.get_tel_ids(["It's a-me, Mario!"])

    # test single string
    assert subarray.get_tel_ids("LST_LST_LSTCam") == (1, 2, 3, 4)

    # test single id
    assert subarray.get_tel_ids(1) == (1,)

    # test invalid id
    with pytest.raises(ValueError):
        subarray.get_tel_ids(500)

    # test invalid description
    with pytest.raises(ValueError):
        subarray.get_tel_ids("It's a-me, Mario!")


def test_unknown_telescopes(example_subarray):
    from ctapipe.instrument import UnknownTelescopeID

    with pytest.raises(UnknownTelescopeID):
        example_subarray.select_subarray([300, 201])


def test_multiplicity(subarray_prod5_paranal):
    subarray = subarray_prod5_paranal.select_subarray([1, 2, 20, 21, 80, 81])

    mask = np.array([True, False, True, True, False, False])

    assert subarray.multiplicity(mask) == 3
    assert subarray.multiplicity(mask, "LST_LST_LSTCam") == 1
    assert subarray.multiplicity(mask, "MST_MST_FlashCam") == 2
    assert subarray.multiplicity(mask, "SST_ASTRI_CHEC") == 0

    masks = np.array(
        [
            [True, False, True, True, False, False],
            [True, True, False, True, False, True],
        ]
    )

    np.testing.assert_equal(subarray.multiplicity(masks), [3, 4])
    np.testing.assert_equal(subarray.multiplicity(masks, "LST_LST_LSTCam"), [1, 2])
    np.testing.assert_equal(subarray.multiplicity(masks, "MST_MST_FlashCam"), [2, 1])
    np.testing.assert_equal(subarray.multiplicity(masks, "SST_ASTRI_CHEC"), [0, 1])


def test_subarrays(subarray_prod5_paranal: SubarrayDescription):
    """
    Check that constructing a new SubarrayDescription by using
    `select_subarray()` works as expected.
    """
    subarray = subarray_prod5_paranal.select_subarray([1, 2, 3, 4], name="NewArray")
    assert subarray.name == "NewArray"
    assert isinstance(subarray.reference_location, EarthLocation)
    assert subarray.reference_location == subarray_prod5_paranal.reference_location
