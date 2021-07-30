import tables


def test_get_structure(dl1_file):
    from ctapipe.io.tableloader import get_structure

    with tables.open_file(dl1_file, "r") as f:
        assert get_structure(f) == "by_id"


def test_load_events(dl1_file):
    from ctapipe.io.tableloader import TableLoader

    with TableLoader(dl1_file) as table_loader:
        table = table_loader.load_telescope_events(tel_id=25)
        assert "hillas_length" in table.colnames
        assert "time" in table.colnames
        assert "event_type" in table.colnames

    with TableLoader(dl1_file, load_dl1_images=True) as table_loader:
        table = table_loader.load_telescope_events(tel_id=25)
        assert "image" in table.colnames

    assert not table_loader.h5file.isopen
