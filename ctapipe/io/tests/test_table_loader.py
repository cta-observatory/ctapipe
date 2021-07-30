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


def test_get_structure(test_file):
    from ctapipe.io.tableloader import get_structure

    expected, path = test_file

    with tables.open_file(path, "r") as f:
        assert get_structure(f) == expected


def test_load_events(test_file):
    from ctapipe.io.tableloader import TableLoader

    _, dl1_file = test_file

    with TableLoader(dl1_file) as table_loader:
        table = table_loader.load_telescope_events(tel_id=25)
        assert "hillas_length" in table.colnames
        assert "time" in table.colnames
        assert "event_type" in table.colnames
        assert np.all(table["tel_id"] == 25)

    with TableLoader(dl1_file, load_dl1_images=True) as table_loader:
        table = table_loader.load_telescope_events(tel_id=25)
        assert "image" in table.colnames
        assert np.all(table["tel_id"] == 25)

    assert not table_loader.h5file.isopen
