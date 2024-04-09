import astropy.units as u
import numpy as np
import pytest
import tables
from astropy.table import Table

from ctapipe.instrument.subarray import SubarrayDescription
from ctapipe.io.astropy_helpers import read_table
from ctapipe.utils.datasets import get_dataset_path


def check_equal_array_event_order(table1, table2):
    """
    Check that values and order of array events is consistent in two tables.

    Works for both tables of subarray events (obs_id, tel_id) and
    tables of telescope events (obs_id, tel_id, event_id) and combinations
    of the two.
    """

    def unique_events(table):
        return np.unique(np.array(table[["obs_id", "event_id"]]), return_index=True)

    unique_events1, indicies1 = unique_events(table1)
    unique_events2, indicies2 = unique_events(table2)

    if len(unique_events1) != len(unique_events2):
        raise ValueError("Tables have different numbers of events")

    if np.any(unique_events1 != unique_events2):
        raise ValueError("Tables have different subarray events")

    # we expect the rank in which the events appear in indices to be the
    # same, that means argsort should produce the same result on both:
    order1 = np.argsort(indicies1)
    order2 = np.argsort(indicies2)

    if np.any(order1 != order2):
        raise ValueError("Tables have subarray events in different order")


def test_check_order():
    with pytest.raises(ValueError, match="Tables have different numbers"):
        check_equal_array_event_order(
            Table({"obs_id": [1, 1, 2, 2, 3], "event_id": [1, 2, 1, 2, 1]}),
            Table({"obs_id": [1, 1, 2, 2], "event_id": [1, 2, 1, 2]}),
        )

    with pytest.raises(ValueError, match="Tables have different subarray events"):
        check_equal_array_event_order(
            Table({"obs_id": [1, 1, 2, 2, 3], "event_id": [1, 2, 1, 2, 1]}),
            Table({"obs_id": [1, 1, 2, 2, 4], "event_id": [1, 2, 1, 2, 1]}),
        )

    with pytest.raises(
        ValueError, match="Tables have subarray events in different order"
    ):
        check_equal_array_event_order(
            Table({"obs_id": [1, 1, 3, 2, 2], "event_id": [1, 2, 1, 1, 2]}),
            Table({"obs_id": [1, 1, 2, 2, 3], "event_id": [1, 2, 1, 2, 1]}),
        )

    check_equal_array_event_order(
        Table({"obs_id": [1, 1, 3, 2, 2], "event_id": [1, 2, 1, 1, 2]}),
        Table({"obs_id": [1, 1, 3, 2, 2], "event_id": [1, 2, 1, 1, 2]}),
    )


def test_telescope_events_for_tel_id(dl1_file):
    """Test loading data for a single telescope"""
    from ctapipe.io.tableloader import TableLoader

    loader = TableLoader(dl1_file)

    with loader as table_loader:
        table = table_loader.read_telescope_events([8])
        assert "hillas_length" in table.colnames
        assert "time" in table.colnames
        assert "event_type" in table.colnames
        assert np.all(table["tel_id"] == 8)

    with TableLoader(dl1_file) as table_loader:
        table = table_loader.read_telescope_events([8], dl1_images=True)
        assert "image" in table.colnames
        assert np.all(table["tel_id"] == 8)
        assert table["obs_id"].dtype == np.int32

    assert not table_loader.h5file.isopen


def test_telescope_muon_events_for_tel_id(dl1_muon_output_file):
    """Test loading muon data for a single telescope"""
    from ctapipe.io.tableloader import TableLoader

    with TableLoader(
        dl1_muon_output_file,
        focal_length_choice="EQUIVALENT",
    ) as table_loader:
        table = table_loader.read_telescope_events(
            [1], dl1_muons=True, dl1_parameters=False
        )
        assert "muonring_radius" in table.colnames
        assert "muonparameters_containment" in table.colnames
        assert "muonefficiency_optical_efficiency" in table.colnames
        assert np.all(table["tel_id"] == 1)

    assert not table_loader.h5file.isopen


def test_instrument(dl1_file):
    """Test joining instrument data onto telescope events"""
    from ctapipe.io.tableloader import TableLoader

    with TableLoader(dl1_file) as table_loader:
        expected = table_loader.subarray.tel[8].optics.equivalent_focal_length
        table = table_loader.read_telescope_events([8], instrument=True)
        assert "equivalent_focal_length" in table.colnames
        assert np.all(table["equivalent_focal_length"] == expected)


def test_simulated(dl1_file):
    """Test joining simulation info onto telescope events"""
    from ctapipe.io.tableloader import TableLoader

    with TableLoader(dl1_file) as table_loader:
        table = table_loader.read_subarray_events()
        assert "true_energy" in table.colnames
        assert table["obs_id"].dtype == np.int32

        table = table_loader.read_telescope_events([8])
        assert "true_energy" in table.colnames
        assert "true_impact_distance" in table.colnames


def test_true_images(dl1_file):
    """Test joining true images onto telescope events"""
    from ctapipe.io.tableloader import TableLoader

    with TableLoader(dl1_file) as table_loader:
        table = table_loader.read_telescope_events(
            ["MST_MST_NectarCam"], dl1_parameters=False, true_images=True
        )
        assert "true_image" in table.colnames


def test_true_parameters(dl1_file):
    """Test joining true parameters onto telescope events"""
    from ctapipe.io.tableloader import TableLoader

    with TableLoader(dl1_file) as table_loader:
        table = table_loader.read_telescope_events(dl1_parameters=False)
        assert "true_hillas_intensity" in table.colnames


def test_observation_info(dl1_file):
    """Test joining observation info onto telescope events"""
    from ctapipe.io.tableloader import TableLoader

    with TableLoader(dl1_file) as table_loader:
        table = table_loader.read_telescope_events(observation_info=True)
        assert "subarray_pointing_lat" in table.colnames


def test_read_subarray_events(dl2_shower_geometry_file):
    """Test reading subarray events"""
    from ctapipe.io.tableloader import TableLoader

    with TableLoader(dl2_shower_geometry_file) as table_loader:
        table = table_loader.read_subarray_events()
        assert "HillasReconstructor_alt" in table.colnames
        assert "true_energy" in table.colnames
        assert "time" in table.colnames


def test_table_loader_keeps_original_order(dl2_merged_file):
    """Test reading subarray events keeps order in file"""
    from ctapipe.io.tableloader import TableLoader

    # check that the order is the same as in the file itself
    trigger = read_table(dl2_merged_file, "/dl1/event/subarray/trigger")
    # check we actually have unsorted input
    assert not np.all(np.diff(trigger["obs_id"]) >= 0)

    with TableLoader(dl2_merged_file) as table_loader:
        events = table_loader.read_subarray_events()
        tel_events = table_loader.read_telescope_events()

    check_equal_array_event_order(events, trigger)
    check_equal_array_event_order(events, tel_events)


def test_read_telescope_events_type(dl2_shower_geometry_file):
    """Test reading telescope events for a given telescope type"""

    from ctapipe.io.tableloader import TableLoader

    subarray = SubarrayDescription.from_hdf(dl2_shower_geometry_file)

    with TableLoader(dl2_shower_geometry_file) as table_loader:
        table = table_loader.read_telescope_events(
            ["MST_MST_FlashCam"],
            dl1_parameters=False,
            true_images=True,
            instrument=True,
        )

        assert "HillasReconstructor_alt" in table.colnames
        assert "true_energy" in table.colnames
        assert "true_image" in table.colnames
        expected_ids = subarray.get_tel_ids_for_type("MST_MST_FlashCam")
        assert set(table["tel_id"].data).issubset(expected_ids)
        assert "equivalent_focal_length" in table.colnames
        # regression test for #2051
        assert "HillasReconstructor_tel_impact_distance" in table.colnames


def test_read_telescope_events_by_type(dl2_shower_geometry_file):
    """Test reading telescope events for by types"""

    from ctapipe.io.tableloader import TableLoader

    subarray = SubarrayDescription.from_hdf(dl2_shower_geometry_file)

    with TableLoader(dl2_shower_geometry_file) as table_loader:
        tables = table_loader.read_telescope_events_by_type(
            [25, 130],
            dl1_images=False,
            dl1_parameters=False,
            true_images=True,
            instrument=True,
        )

        for tel_type in ["MST_MST_NectarCam", "MST_MST_FlashCam"]:
            table = tables[tel_type]

            assert "HillasReconstructor_alt" in table.colnames
            assert "true_energy" in table.colnames
            assert "true_image" in table.colnames
            expected_ids = subarray.get_tel_ids_for_type(tel_type)
            assert set(table["tel_id"].data).issubset(expected_ids)
            assert "equivalent_focal_length" in table.colnames


def test_h5file(dl2_shower_geometry_file):
    """Test we can also pass an already open h5file"""
    from ctapipe.io.tableloader import TableLoader

    # no input raises error
    with pytest.raises(ValueError):
        with TableLoader():
            pass

    # test we can use an already open file
    with tables.open_file(dl2_shower_geometry_file, mode="r+") as h5file:
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
    start = 0
    stop = chunk_size

    with TableLoader(dl2_shower_geometry_file) as table_loader:
        tel_event_it = table_loader.read_telescope_events_chunked(
            chunk_size=chunk_size, true_parameters=False
        )
        event_it = table_loader.read_subarray_events_chunked(chunk_size=chunk_size)
        by_type_it = table_loader.read_telescope_events_by_type_chunked(
            chunk_size=chunk_size, true_parameters=False
        )
        by_id_it = table_loader.read_telescope_events_by_id_chunked(
            chunk_size=chunk_size, true_parameters=False
        )

        iters = (event_it, tel_event_it, by_type_it, by_id_it)

        for chunk, (events, tel_events, by_type, by_id) in enumerate(zip(*iters)):
            expected_start = chunk * chunk_size
            expected_stop = min(n_events, (chunk + 1) * chunk_size)

            start, stop, events = events
            tel_events = tel_events.data
            by_type = by_type.data
            by_id = by_id.data
            assert expected_start == start
            assert expected_stop == stop

            n_read += len(events)

            # last chunk might be smaller
            if chunk == (n_chunks - 1):
                assert len(events) == n_events % chunk_size
            else:
                assert len(events) == chunk_size

            # check events are in compatible order
            check_equal_array_event_order(events, tel_events)
            check_equal_array_event_order(trigger[start:stop], events)
            np.testing.assert_allclose(
                tel_events["telescope_pointing_altitude"].quantity.to_value(u.deg), 70
            )
            np.testing.assert_allclose(
                tel_events["telescope_pointing_azimuth"].quantity.to_value(u.deg), 0
            )

            # check number of telescope events is correct
            assert len(tel_events) == np.count_nonzero(events["tels_with_trigger"])

            n_events_by_type = 0
            for table in by_type.values():
                n_events_by_type += len(table)
                assert set(zip(table["obs_id"], table["event_id"])).issubset(
                    set(
                        zip(
                            trigger[start:stop]["obs_id"],
                            trigger[start:stop]["event_id"],
                        )
                    )
                )
                assert not np.ma.is_masked(table["HillasReconstructor_is_valid"])
            assert n_events_by_type == len(tel_events)

            n_events_by_id = 0
            for table in by_id.values():
                n_events_by_id += len(table)
                assert set(zip(table["obs_id"], table["event_id"])).issubset(
                    set(
                        zip(
                            trigger[start:stop]["obs_id"],
                            trigger[start:stop]["event_id"],
                        )
                    )
                )
                assert not np.ma.is_masked(table["HillasReconstructor_is_valid"])
            assert n_events_by_type == len(tel_events)

    assert n_read == n_events


def test_read_simulation_config(dl2_merged_file):
    from ctapipe.io import TableLoader

    with TableLoader(dl2_merged_file) as table_loader:
        runs = table_loader.read_simulation_configuration()
        assert len(runs) == 2
        assert np.all(runs["obs_id"] == [4, 1])
        assert u.allclose(runs["energy_range_min"].quantity, [0.004, 0.003] * u.TeV)


def test_read_shower_distributions(dl2_merged_file):
    from ctapipe.io import TableLoader

    with TableLoader(dl2_merged_file) as table_loader:
        histograms = table_loader.read_shower_distribution()
        assert len(histograms) == 2
        assert np.all(histograms["obs_id"] == [4, 1])
        assert np.all(histograms["n_entries"] == [2000, 1000])
        assert np.all(histograms["histogram"].sum(axis=(1, 2)) == [2000, 1000])


def test_read_unavailable_telescope(dl2_shower_geometry_file):
    """Reading a telescope that is not part of the subarray of the file should fail."""
    from ctapipe.io import TableLoader

    with TableLoader(dl2_shower_geometry_file) as loader:
        tel_id = max(loader.subarray.tel.keys()) + 1
        with pytest.raises(ValueError):
            loader.read_telescope_events(
                [tel_id],
                dl1_parameters=False,
            )


def test_read_empty_table(dl2_shower_geometry_file):
    """Reading an empty table should return an empty table."""
    from ctapipe.io import TableLoader

    with TableLoader(dl2_shower_geometry_file) as loader:
        table = loader.read_telescope_events(
            [6],
            dl1_parameters=False,
        )
        assert len(table) == 0


def test_order_merged():
    """Test reading functions return data in correct event order"""
    from ctapipe.io import TableLoader

    path = get_dataset_path("gamma_diffuse_dl2_train_small.dl2.h5")

    trigger = read_table(path, "/dl1/event/subarray/trigger")
    tel_trigger = read_table(path, "/dl1/event/telescope/trigger")
    with TableLoader(path) as loader:
        events = loader.read_subarray_events(dl2=True, observation_info=True)
        check_equal_array_event_order(events, trigger)

        tables = loader.read_telescope_events_by_id()

        for tel_id, table in tables.items():
            mask = tel_trigger["tel_id"] == tel_id
            check_equal_array_event_order(table, tel_trigger[mask])

        tables = loader.read_telescope_events_by_type()
        for tel, table in tables.items():
            mask = np.isin(tel_trigger["tel_id"], loader.subarray.get_tel_ids(tel))
            check_equal_array_event_order(table, tel_trigger[mask])


def test_interpolate_pointing(dl1_mon_pointing_file, tmp_path):
    from ctapipe.io import TableLoader

    with TableLoader(dl1_mon_pointing_file, pointing=True) as loader:
        events = loader.read_telescope_events([4])
        assert len(events) > 0
        assert "telescope_pointing_azimuth" in events.colnames
        assert "telescope_pointing_altitude" in events.colnames
