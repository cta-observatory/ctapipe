import tables
from astropy.table import vstack

from ctapipe.instrument.subarray import SubarrayDescription
from ctapipe.io.astropy_helpers import read_table
from ctapipe.io.tests.test_astropy_helpers import assert_table_equal


def compare_table(in1, in2, merged, table):
    t1 = read_table(in1, table) if table in in1 else None
    t2 = read_table(in2, table) if table in in2 else None

    if t1 is None and t2 is None:
        return 0

    if t1 is not None and t2 is not None:
        stacked = vstack([t1, t2])
    elif t1 is None:
        stacked = t2
    else:
        stacked = t1

    assert_table_equal(stacked, read_table(merged, table))
    return 1


def test_simple(tmp_path, dl1_file, dl1_proton_file):
    from ctapipe.io.select_merge_hdf5 import SelectMergeHDF5

    output = tmp_path / "merged_simple.dl1.h5"

    with SelectMergeHDF5(output) as merger:
        merger(dl1_file)
        merger(dl1_proton_file)

    subarray = SubarrayDescription.from_hdf(dl1_file)
    assert subarray == SubarrayDescription.from_hdf(output), "Subarays do not match"

    tel_groups = [
        "/dl1/event/telescope/parameters",
        "/dl1/event/telescope/images",
        "/simulation/event/telescope/parameters",
        "/simulation/event/telescope/images",
    ]

    tables_checked = 0
    tables_to_check = [
        "/dl1/event/telescope/trigger",
        "/dl1/event/subarray/trigger",
    ]
    for group in tel_groups:
        for tel_id in subarray.tel:
            table = f"{group}/tel_{tel_id:03d}"
            tables_to_check.append(table)

    with (
        tables.open_file(dl1_file) as in1,
        tables.open_file(dl1_proton_file) as in2,
        tables.open_file(output) as merged,
    ):

        for table in tables_to_check:
            tables_checked += compare_table(in1, in2, merged, table)

        print(f"Checked {tables_checked} tables")
        assert tables_checked > 0
