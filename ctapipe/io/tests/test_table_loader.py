import pytest
import tables
import numpy as np

from ctapipe.instrument.subarray import SubarrayDescription


@pytest.fixture(params=["by_type", "by_id"])
def test_file(request, dl1_file, dl1_by_type_file):
    """Test dl1 files in both structures"""
    if request.param == "by_type":
        f = dl1_by_type_file
    else:
        f = dl1_file

    return request.param, f


@pytest.fixture(params=["by_type", "by_id"])
def test_file_dl2(request, dl2_shower_geometry_file, dl2_shower_geometry_file_type):
    """Test dl2 files in both structures"""
    if request.param == "by_type":
        f = dl2_shower_geometry_file
    else:
        f = dl2_shower_geometry_file_type

    return request.param, f


def test_get_structure(test_file):
    """Test _get_structure"""
    from ctapipe.io.tableloader import _get_structure

    expected, path = test_file

    with tables.open_file(path, "r") as f:
        assert _get_structure(f) == expected


def test_telescope_events_for_tel_id(test_file):
    """Test loading data for a single telescope"""
    from ctapipe.io.tableloader import TableLoader

    _, dl1_file = test_file

    loader = TableLoader(dl1_file, load_dl1_parameters=True, load_trigger=True)

    with loader as table_loader:
        table = table_loader.read_telescope_events([8])
        assert "hillas_length" in table.colnames
        assert "time" in table.colnames
        assert "event_type" in table.colnames
        assert np.all(table["tel_id"] == 8)

    with TableLoader(dl1_file, load_dl1_images=True) as table_loader:
        table = table_loader.read_telescope_events([8])
        assert "image" in table.colnames
        assert np.all(table["tel_id"] == 8)
        assert table["obs_id"].dtype == np.int32

    assert not table_loader.h5file.isopen


def test_load_instrument(test_file):
    """Test joining instrument data onto telescope events"""
    from ctapipe.io.tableloader import TableLoader

    _, dl1_file = test_file

    with TableLoader(dl1_file, load_instrument=True) as table_loader:
        expected = table_loader.subarray.tel[8].optics.equivalent_focal_length
        table = table_loader.read_telescope_events([8])
        assert "equivalent_focal_length" in table.colnames
        assert np.all(table["equivalent_focal_length"] == expected)


def test_load_simulated(test_file):
    """Test joining simulation info onto telescope events"""
    from ctapipe.io.tableloader import TableLoader

    _, dl1_file = test_file

    with TableLoader(dl1_file, load_simulated=True) as table_loader:
        table = table_loader.read_subarray_events()
        assert "true_energy" in table.colnames
        assert table["obs_id"].dtype == np.int32

        table = table_loader.read_telescope_events([8])
        assert "true_energy" in table.colnames


def test_true_images(test_file):
    """Test joining true images onto telescope events"""
    from ctapipe.io.tableloader import TableLoader

    _, dl1_file = test_file

    with TableLoader(
        dl1_file, load_dl1_parameters=False, load_true_images=True
    ) as table_loader:
        table = table_loader.read_telescope_events(["MST_MST_NectarCam"])
        assert "true_image" in table.colnames


def test_true_parameters(test_file):
    """Test joining true parameters onto telescope events"""
    from ctapipe.io.tableloader import TableLoader

    _, dl1_file = test_file

    with TableLoader(
        dl1_file, load_dl1_parameters=False, load_true_parameters=True
    ) as table_loader:
        table = table_loader.read_telescope_events()
        assert "true_hillas_intensity" in table.colnames


def test_read_subarray_events(test_file_dl2):
    """Test reading subarray events"""

    from ctapipe.io.tableloader import TableLoader

    _, dl2_file = test_file_dl2

    with TableLoader(
        dl2_file, load_dl2=True, load_simulated=True, load_trigger=True
    ) as table_loader:
        table = table_loader.read_subarray_events()
        assert "HillasReconstructor_alt" in table.colnames
        assert "true_energy" in table.colnames
        assert "time" in table.colnames


def test_read_telescope_events_type(test_file_dl2):
    """Test reading telescope events for a given telescope type"""

    from ctapipe.io.tableloader import TableLoader

    _, dl2_file = test_file_dl2

    subarray = SubarrayDescription.from_hdf(dl2_file)

    with TableLoader(
        dl2_file,
        load_dl1_images=False,
        load_dl1_parameters=False,
        load_dl2=True,
        load_simulated=True,
        load_true_images=True,
        load_trigger=False,
        load_instrument=True,
    ) as table_loader:

        table = table_loader.read_telescope_events(["MST_MST_FlashCam"])

        assert "HillasReconstructor_alt" in table.colnames
        assert "true_energy" in table.colnames
        assert "true_image" in table.colnames
        expected_ids = subarray.get_tel_ids_for_type("MST_MST_FlashCam")
        assert set(table["tel_id"].data).issubset(expected_ids)
        assert "equivalent_focal_length" in table.colnames


def test_read_telescope_events_by_type(test_file_dl2):
    """Test reading telescope events for by types"""

    from ctapipe.io.tableloader import TableLoader

    _, dl2_file = test_file_dl2
    subarray = SubarrayDescription.from_hdf(dl2_file)

    with TableLoader(
        dl2_file,
        load_dl1_images=False,
        load_dl1_parameters=False,
        load_dl2=True,
        load_simulated=True,
        load_true_images=True,
        load_trigger=False,
        load_instrument=True,
    ) as table_loader:

        tables = table_loader.read_telescope_events_by_type([25, 130])

        for tel_type in ["MST_MST_NectarCam", "MST_MST_FlashCam"]:

            table = tables[tel_type]

            assert "HillasReconstructor_alt" in table.colnames
            assert "true_energy" in table.colnames
            assert "true_image" in table.colnames
            expected_ids = subarray.get_tel_ids_for_type(tel_type)
            assert set(table["tel_id"].data).issubset(expected_ids)
            assert "equivalent_focal_length" in table.colnames


def test_h5file(test_file_dl2):
    """Test we can also pass an already open h5file"""
    from ctapipe.io.tableloader import TableLoader

    _, dl2_file = test_file_dl2

    # no input raises error
    with pytest.raises(ValueError):
        with TableLoader():
            pass

    # test we can use an already open file
    with tables.open_file(dl2_file, mode="r+") as h5file:
        with TableLoader(h5file=h5file) as loader:
            assert 25 in loader.subarray.tel
            loader.read_subarray_events()
            loader.read_telescope_events()


def test_chunked(dl2_shower_geometry_file):
    """Test chunked reading"""
    from ctapipe.io.tableloader import TableLoader, read_table

    trigger = read_table(dl2_shower_geometry_file, "/dl1/event/subarray/trigger")
    n_events = len(trigger)
    n_read = 0

    n_chunks = 2
    chunk_size = int(np.ceil(n_events / n_chunks))

    with TableLoader(
        dl2_shower_geometry_file,
        load_dl1_images=False,
        load_true_images=False,
        load_dl1_parameters=True,
        load_dl2=True,
        load_simulated=True,
        load_trigger=True,
    ) as table_loader:

        tel_event_it = table_loader.read_telescope_events_chunked(chunk_size)
        event_it = table_loader.read_subarray_events_chunked(chunk_size)

        for chunk, (events, tel_events) in enumerate(zip(event_it, tel_event_it)):
            n_read += len(events)

            # last chunk might be smaller
            if chunk == (n_chunks - 1):
                assert len(events) == n_events % chunk_size
            else:
                assert len(events) == chunk_size

            assert len(tel_events) == np.sum(events["tels_with_trigger"])

            unique_events_from_tel_table = np.unique(tel_events[["obs_id", "event_id"]])

            # test we get matching obs_ids / event_ids
            idx = events[["obs_id", "event_id"]].argsort()
            sorted_index = np.array(events[["obs_id", "event_id"]][idx])
            assert (unique_events_from_tel_table == sorted_index).all()

    assert n_read == n_events
