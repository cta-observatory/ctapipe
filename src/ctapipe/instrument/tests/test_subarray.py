"""Tests for SubarrayDescriptions"""

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
    sub_meta_hdf_table = sub.to_table(kind="subarray")
    sub_meta_fits_table = sub.to_table(kind="subarray", meta_convention="fits")

    assert len(sub_meta_hdf_table) == sub.n_tels
    assert len(sub.to_table(kind="optics")) == len(sub.optics_types)
    assert "OBSGEO-X" in sub_meta_fits_table.meta
    assert "reference_itrs_x" in sub_meta_hdf_table.meta

    with pytest.raises(ValueError):
        sub.to_table(kind="NON_EXISTENT_KIND")
    with pytest.raises(ValueError):
        sub.to_table(meta_convention="NON_EXISTENT_META_CONVENTION")


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


def test_check_matchings_subarray(example_subarray, subarray_prod5_paranal):
    """Test SubarrayDescription.check_matching_subarrays static method"""
    assert SubarrayDescription.check_matching_subarrays(
        [example_subarray, example_subarray]
    )
    assert not SubarrayDescription.check_matching_subarrays(
        [example_subarray, subarray_prod5_paranal]
    )


def test_tel_earth_locations(example_subarray):
    """Test cached tel_earth_locations property"""
    # Get the cached property
    earth_locs = example_subarray.tel_earth_locations

    assert isinstance(earth_locs, dict)
    assert len(earth_locs) == example_subarray.n_tels

    # Check all telescope IDs are present
    for tel_id in example_subarray.tel_ids:
        assert tel_id in earth_locs
        assert isinstance(earth_locs[tel_id], EarthLocation)

    # Verify conversion is correct by comparing with manual conversion
    tel_id = example_subarray.tel_ids[0]
    tel_index = example_subarray.tel_index_array[tel_id]
    manual_location = example_subarray.tel_coords[tel_index].to_earth_location()

    assert u.isclose(earth_locs[tel_id].x, manual_location.x)
    assert u.isclose(earth_locs[tel_id].y, manual_location.y)
    assert u.isclose(earth_locs[tel_id].z, manual_location.z)

    # Verify it's cached (same object returned)
    earth_locs_2 = example_subarray.tel_earth_locations
    assert earth_locs is earth_locs_2


def test_load_array_element_ids():
    """Test loading array element IDs from service data"""
    import warnings

    from ctapipe.core.provenance import MissingReferenceMetadata

    # Suppress expected warning about missing reference metadata in service data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=MissingReferenceMetadata)
        data = SubarrayDescription.load_array_element_ids()

    assert isinstance(data, dict)
    assert "metadata" in data
    assert "array_elements" in data

    array_elements = data["array_elements"]
    assert isinstance(array_elements, list)
    assert len(array_elements) > 0

    # Check structure of first element
    first_element = array_elements[0]
    assert "id" in first_element
    assert "name" in first_element
    assert isinstance(first_element["id"], int)
    assert isinstance(first_element["name"], str)


def test_load_array_element_positions():
    """Test loading array element positions from service data"""
    # Test for CTAO North
    positions_n = SubarrayDescription.load_array_element_positions("ctao_n")

    assert "ae_id" in positions_n.colnames
    assert "name" in positions_n.colnames
    assert "x" in positions_n.colnames
    assert "y" in positions_n.colnames
    assert "z" in positions_n.colnames

    # Check metadata contains reference location
    assert "reference_x" in positions_n.meta
    assert "reference_y" in positions_n.meta
    assert "reference_z" in positions_n.meta

    # Check some positions are valid
    assert len(positions_n) > 0
    assert positions_n["x"].unit == u.m
    assert positions_n["y"].unit == u.m
    assert positions_n["z"].unit == u.m


def test_load_subarray_info():
    """Test loading subarray definitions from service data"""
    import warnings

    from ctapipe.core.provenance import MissingReferenceMetadata

    # Suppress expected warning about missing reference metadata in service data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=MissingReferenceMetadata)
        # Load all subarray info
        all_data = SubarrayDescription.load_subarray_info()

        assert isinstance(all_data, dict)
        assert "metadata" in all_data
        assert "subarrays" in all_data

        subarrays = all_data["subarrays"]
        assert isinstance(subarrays, list)
        assert len(subarrays) > 0

        # Check structure of first subarray
        first_subarray = subarrays[0]
        assert "id" in first_subarray
        assert "name" in first_subarray
        assert "array_element_ids" in first_subarray
        assert "site" in first_subarray

        # Load specific subarray by ID
        subarray_id = first_subarray["id"]
        specific = SubarrayDescription.load_subarray_info(subarray_id)

        assert specific["id"] == subarray_id
        assert specific["name"] == first_subarray["name"]

        # Test with invalid ID
        with pytest.raises(ValueError, match="Subarray ID .* not found"):
            SubarrayDescription.load_subarray_info(99999)


def test_from_service_data_minimal(svc_path):
    """Test creating SubarrayDescription from service data with minimal parameters"""
    import warnings

    from ctapipe.core.provenance import MissingReferenceMetadata
    from ctapipe.instrument.warnings import FromNameWarning

    # Use subarray ID 1 (LST1) which should exist in the test data
    # Suppress expected warnings about service data files and from_name usage
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=MissingReferenceMetadata)
        warnings.filterwarnings("ignore", category=FromNameWarning)
        subarray = SubarrayDescription.from_service_data(subarray_id=1, site="ctao_n")

    assert isinstance(subarray, SubarrayDescription)
    assert subarray.name == "LST1"  # From the JSON data
    assert subarray.n_tels >= 1
    assert isinstance(subarray.reference_location, EarthLocation)

    # Check that telescopes have proper descriptions
    for tel_id, tel_desc in subarray.tel.items():
        assert isinstance(tel_desc, TelescopeDescription)
        assert tel_desc.camera is not None
        assert tel_desc.optics is not None


def test_from_service_data_with_camera_optics_names(svc_path):
    """Test creating SubarrayDescription with explicit camera and optics names"""
    import warnings

    from ctapipe.core.provenance import MissingReferenceMetadata
    from ctapipe.instrument.warnings import FromNameWarning

    subarray_id = 2  # CTAO-N LSTs

    # Suppress expected warnings about service data and from_name usage
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FromNameWarning)
        warnings.filterwarnings("ignore", category=MissingReferenceMetadata)

        # Get the array element IDs first
        info = SubarrayDescription.load_subarray_info(subarray_id)
        ae_ids = info["array_element_ids"]

        # Create camera and optics mappings
        camera_names = {ae_id: "LSTCam" for ae_id in ae_ids}
        optics_names = {ae_id: "LST" for ae_id in ae_ids}
        subarray = SubarrayDescription.from_service_data(
            subarray_id=subarray_id,
            site="ctao_n",
            camera_names=camera_names,
            optics_names=optics_names,
        )

    assert isinstance(subarray, SubarrayDescription)
    assert subarray.n_tels == len(ae_ids)

    # Check all telescopes have LSTCam camera and LST optics
    for tel_desc in subarray.tel.values():
        assert tel_desc.camera.name == "LSTCam"
        assert tel_desc.optics.name == "LST"


def test_infer_camera_names_north():
    """Test camera name inference for CTAO North"""
    tel_positions = {1: u.Quantity([0, 0, 0], u.m)}
    ae_id_to_name = {1: "LSTN-01"}

    camera_names = SubarrayDescription._infer_camera_names(
        tel_positions, ae_id_to_name, "ctao_n"
    )
    assert camera_names[1] == "LSTCam"

    # Test MST North
    ae_id_to_name = {1: "MSTN-01"}
    camera_names = SubarrayDescription._infer_camera_names(
        tel_positions, ae_id_to_name, "ctao_n"
    )
    assert camera_names[1] == "NectarCam"

    # Test SST
    ae_id_to_name = {1: "SSTS-01"}
    camera_names = SubarrayDescription._infer_camera_names(
        tel_positions, ae_id_to_name, "ctao_n"
    )
    assert camera_names[1] == "SSTCam"


def test_infer_camera_names_south():
    """Test camera name inference for CTAO South"""
    tel_positions = {1: u.Quantity([0, 0, 0], u.m)}
    ae_id_to_name = {1: "MSTS-01"}

    camera_names = SubarrayDescription._infer_camera_names(
        tel_positions, ae_id_to_name, "ctao_s"
    )
    assert camera_names[1] == "FlashCam"


def test_infer_optics_names():
    """Test optics name inference from telescope names"""
    tel_positions = {1: u.Quantity([0, 0, 0], u.m), 2: u.Quantity([1, 1, 1], u.m)}
    ae_id_to_name = {1: "LSTN-01", 2: "MSTN-01"}

    optics_names = SubarrayDescription._infer_optics_names(tel_positions, ae_id_to_name)

    assert optics_names[1] == "LST"
    assert optics_names[2] == "MST"


def test_build_tel_positions():
    """Test building telescope positions from table"""
    import warnings

    from astropy.table import Table

    positions_table = Table(
        {
            "ae_id": [1, 2, 3],
            "x": [100.0, 200.0, np.nan] * u.m,
            "y": [150.0, 250.0, np.nan] * u.m,
            "z": [2000.0, 2010.0, np.nan] * u.m,
        }
    )

    array_element_ids = [1, 2, 3, 4]  # 3 has NaN, 4 doesn't exist

    # Suppress expected warnings about NaN and missing positions
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Array element .* has NaN position")
        warnings.filterwarnings("ignore", message="Array element .* not found")
        tel_positions = SubarrayDescription._build_tel_positions(
            positions_table, array_element_ids
        )

    # Should only have positions for 1 and 2 (3 has NaN, 4 doesn't exist)
    assert len(tel_positions) == 2
    assert 1 in tel_positions
    assert 2 in tel_positions
    assert 3 not in tel_positions  # NaN position
    assert 4 not in tel_positions  # Not in table

    # Check values
    assert np.allclose(tel_positions[1].value, [100.0, 150.0, 2000.0])
    assert np.allclose(tel_positions[2].value, [200.0, 250.0, 2010.0])
