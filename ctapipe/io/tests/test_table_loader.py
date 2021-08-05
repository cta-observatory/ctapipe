import pytest
import tables
import numpy as np


@pytest.fixture(params=["by_type", "by_id"])
def test_file(request, dl1_file, dl1_by_type_file):
    if request.param == "by_type":
        f = dl1_by_type_file
    else:
        f = dl1_file

    return request.param, f


@pytest.fixture(params=["by_type", "by_id"])
def test_file_dl2(request, dl2_shower_geometry_file, dl2_shower_geometry_file_type):
    if request.param == "by_type":
        f = dl2_shower_geometry_file
    else:
        f = dl2_shower_geometry_file_type

    return request.param, f


def test_get_tel_ids(test_file):

    from ctapipe.io.tableloader import get_tel_ids
    from ctapipe.instrument import SubarrayDescription
    from ctapipe.instrument import TelescopeDescription

    _, dl1_file = test_file

    sst = TelescopeDescription(tel_type="SST", name="ASTRI",
                               optics="ASTRI", camera="CHEC")

    subarray = SubarrayDescription.from_hdf(dl1_file)

    labels = [1, 2, "MST_MST_FlashCam", sst]

    tel_ids = get_tel_ids(subarray, labels)

    true_tel_ids = (subarray.get_tel_ids_for_type("MST_MST_FlashCam")
                    + subarray.get_tel_ids_for_type(sst)
                    + [1, 2])

    assert sorted(tel_ids) == sorted(true_tel_ids)

    # test invalid telescope type
    with pytest.raises(Exception):
        tel_ids = get_tel_ids(subarray, ["It's a-me, Mario!"])


def test_get_structure(test_file):
    from ctapipe.io.tableloader import get_structure

    expected, path = test_file

    with tables.open_file(path, "r") as f:
        assert get_structure(f) == expected


def test_read_events_for_tel_id(test_file):
    from ctapipe.io.tableloader import TableLoader

    _, dl1_file = test_file

    loader = TableLoader(dl1_file,
                         load_dl1_parameters=True,
                         load_trigger=True)

    with loader as table_loader:
        table = table_loader.read_events([25])
        assert "hillas_length" in table.colnames
        assert "time" in table.colnames
        assert "event_type" in table.colnames
        assert np.all(table["tel_id"] == 25)

    with TableLoader(dl1_file, load_dl1_images=True) as table_loader:
        table = table_loader.read_events([25])
        assert "image" in table.colnames
        assert np.all(table["tel_id"] == 25)

    assert not table_loader.h5file.isopen


def test_load_instrument(test_file):
    from ctapipe.io.tableloader import TableLoader

    _, dl1_file = test_file

    with TableLoader(dl1_file, load_instrument=True) as table_loader:
        expected = table_loader.subarray.tel[25].optics.equivalent_focal_length
        table = table_loader.read_telescope_events_for_id(tel_id=25)
        assert "equivalent_focal_length" in table.colnames
        assert np.all(table["equivalent_focal_length"] == expected)


def test_load_simulated(test_file):
    from ctapipe.io.tableloader import TableLoader

    _, dl1_file = test_file

    with TableLoader(dl1_file, load_simulated=True) as table_loader:
        table = table_loader.read_events([25])
        assert "true_energy" in table.colnames


def test_true_images(test_file):
    from ctapipe.io.tableloader import TableLoader

    _, dl1_file = test_file

    with TableLoader(
        dl1_file, load_dl1_parameters=False, load_true_images=True
    ) as table_loader:
        table = table_loader.read_telescope_events_for_id(tel_id=25)
        assert "true_image" in table.colnames


def test_read_subarray_events(test_file_dl2):

    from ctapipe.io.tableloader import TableLoader

    _, dl2_file = test_file_dl2

    with TableLoader(
        dl2_file,
        load_dl1_images=False,
        load_dl1_parameters=False,
        load_dl2_geometry=True,
        load_simulated=True,
        load_true_images=False,
        load_trigger=True,
        load_instrument=False,
    ) as table_loader:
        table = table_loader.read_subarray_events()
        assert "HillasReconstructor_alt" in table.colnames
        assert "true_energy" in table.colnames


def test_read_events(test_file_dl2):

    from ctapipe.io.tableloader import TableLoader

    _, dl2_file = test_file_dl2

    with TableLoader(
        dl2_file,
        load_dl1_images=False,
        load_dl1_parameters=False,
        load_dl2_geometry=True,
        load_simulated=True,
        load_true_images=True,
        load_trigger=False,
        load_instrument=True,
    ) as table_loader:

        table = table_loader.read_events(["MST_MST_FlashCam"])

        assert "HillasReconstructor_alt" in table.colnames
        assert "true_energy" in table.colnames
        assert "true_image" in table.colnames
        assert set(table["tel_id"].data).issubset([25, 125, 130])
        assert "equivalent_focal_length" in table.colnames


def test_read_events_by_tel_type(test_file_dl2):

    from ctapipe.io.tableloader import TableLoader

    _, dl2_file = test_file_dl2

    with TableLoader(
        dl2_file,
        load_dl1_images=False,
        load_dl1_parameters=False,
        load_dl2_geometry=True,
        load_simulated=True,
        load_true_images=True,
        load_trigger=False,
        load_instrument=True,
    ) as table_loader:

        tables = table_loader.read_events_by_tel_type([25, 130])

        for tel_type in ["MST_MST_NectarCam", "MST_MST_FlashCam"]:

            table = tables[tel_type]

            assert "HillasReconstructor_alt" in table.colnames
            assert "true_energy" in table.colnames
            assert "true_image" in table.colnames
            assert set(table["tel_id"].data).issubset([25, 125, 130])
            assert "equivalent_focal_length" in table.colnames


@pytest.mark.parametrize(
    "telescope_description", ["MST_MST_NectarCam", "MST_MST_FlashCam"]
)
def test_read_events_for_type(telescope_description, test_file):
    from ctapipe.io.tableloader import TableLoader

    _, dl1_file = test_file

    with TableLoader(dl1_file, load_instrument=True) as table_loader:
        table = table_loader.read_events_for_type(telescope_description)
        assert np.all(table["tel_description"] == telescope_description)
