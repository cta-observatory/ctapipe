from itertools import combinations


def test_eq(example_subarray):
    for tel1, tel2 in combinations(example_subarray.tel.values(), 2):
        if tel1.camera.camera_name == tel2.camera.camera_name:
            assert tel1.camera.readout == tel2.camera.readout
        else:
            assert tel1.camera.readout != tel2.camera.readout


def test_hash(example_subarray):
    for tel1, tel2 in combinations(example_subarray.tel.values(), 2):
        if tel1.camera.camera_name == tel2.camera.camera_name:
            assert hash(tel1.camera.readout) == hash(tel2.camera.readout)
        else:
            assert hash(tel1.camera.readout) != hash(tel2.camera.readout)


def test_table_roundtrip(prod5_lst):
    """Test we can roundtrip using from_table / to_table"""
    from ctapipe.instrument import CameraReadout

    camera_in = prod5_lst.camera.readout

    table = camera_in.to_table()
    assert table.meta["TAB_VER"] == CameraReadout.CURRENT_TAB_VERSION
    assert table.meta["TAB_TYPE"] == "ctapipe.instrument.CameraReadout"
    camera_out = CameraReadout.from_table(table)

    assert camera_in == camera_out
    assert hash(camera_in) == hash(camera_out)
