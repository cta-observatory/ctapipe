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


def compare_stats_table(in1, in2, merged, table, required=True):
    t1 = read_table(in1, table) if table in in1 else None
    t2 = read_table(in2, table) if table in in2 else None

    if required and t1 is None:
        raise ValueError(f" table {table} not present in {in1.filename}")

    if required and t2 is None:
        raise ValueError(f" table {table} not present in {in1.filename}")

    if t1 is None and t2 is None:
        return 0

    if t1 is not None and t2 is not None:
        stacked = t1.copy()
        for col in ("counts", "cumulative_counts"):
            stacked[col] = t1[col] + t2[col]
    elif t1 is None:
        stacked = t2
    else:
        stacked = t1

    assert_table_equal(stacked, read_table(merged, table))
    return 1


def test_simple(tmp_path, gamma_train_clf, proton_train_clf):
    from ctapipe.io.select_merge_hdf5 import HDF5Merger

    output = tmp_path / "merged_simple.dl1.h5"

    with HDF5Merger(output) as merger:
        merger(gamma_train_clf)
        merger(proton_train_clf)

    subarray = SubarrayDescription.from_hdf(gamma_train_clf)
    assert subarray == SubarrayDescription.from_hdf(output), "Subarays do not match"

    tel_groups = [
        "/dl1/event/telescope/parameters",
        "/dl1/event/telescope/images",
        "/dl2/event/telescope/impact/HillasReconstructor",
        "/dl2/event/telescope/energy/ExtraTreesRegressor",
        "/simulation/event/telescope/parameters",
        "/simulation/event/telescope/images",
        "/simulation/event/telescope/impact",
    ]

    tables_checked = 0
    tables_to_check = [
        "/dl2/event/subaray/energy/ExtraTreesRegressor",
        "/dl2/event/subarray/geometry/HillasReconstructor",
        "/dl1/event/telescope/trigger",
        "/dl1/event/subarray/trigger",
        "/simulation/event/subarray/shower",
        "/simulation/service/shower_distribution",
    ]
    for group in tel_groups:
        for tel_id in subarray.tel:
            table = f"{group}/tel_{tel_id:03d}"
            tables_to_check.append(table)

    statistics_tables = [
        "/dl1/service/image_statistics",
        "/dl2/service/tel_event_statistics/HillasReconstructor",
    ]

    with (
        tables.open_file(gamma_train_clf) as in1,
        tables.open_file(proton_train_clf) as in2,
        tables.open_file(output) as merged,
    ):

        for table in tables_to_check:
            tables_checked += compare_table(in1, in2, merged, table)

        print(f"Checked {tables_checked} tables")
        assert tables_checked > 0

        for table in statistics_tables:
            tables_checked += compare_stats_table(
                in1, in2, merged, table, required=True
            )
