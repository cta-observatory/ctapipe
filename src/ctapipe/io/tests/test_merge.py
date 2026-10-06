from types import SimpleNamespace

import pytest
import tables
from astropy.table import vstack
from astropy.utils.data import shutil
from ctao_datamodel.models import dataproducts as dp
from traitlets import TraitError
from traitlets.config import Config

from ctapipe.instrument.subarray import SubarrayDescription
from ctapipe.io import metadata
from ctapipe.io.astropy_helpers import read_table
from ctapipe.io.tests.test_astropy_helpers import assert_table_equal
from ctapipe.utils.datasets import get_dataset_path


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


def compare_stats_table(in1, in2, merged, table):
    t1 = read_table(in1, table) if table in in1 else None
    t2 = read_table(in2, table) if table in in2 else None

    if t1 is None:
        raise ValueError(f"table {table} not present in {in1.filename}")

    if t2 is None:
        raise ValueError(f"table {table} not present in {in2.filename}")

    stacked = t1.copy()
    for col in ("counts", "cumulative_counts"):
        stacked[col] = t1[col] + t2[col]

    assert_table_equal(stacked, read_table(merged, table))


def test_split_h5path():
    from ctapipe.io.hdf5merger import split_h5path

    assert split_h5path("/") == ("/", "")
    assert split_h5path("/foo") == ("/", "foo")
    assert split_h5path("/foo/") == ("/", "foo")
    assert split_h5path("/foo/bar") == ("/foo", "bar")
    assert split_h5path("/foo/bar/") == ("/foo", "bar")
    assert split_h5path("/foo/bar/baz") == ("/foo/bar", "baz")

    with pytest.raises(ValueError, match="Path must start with /"):
        split_h5path("foo/bar")


def test_simple(tmp_path, gamma_train_clf, proton_train_clf):
    from ctapipe.io.hdf5merger import HDF5Merger

    output = tmp_path / "merged_simple.dl1.h5"

    with HDF5Merger(output) as merger:
        merger(gamma_train_clf)
        merger(proton_train_clf)

    subarray = SubarrayDescription.from_hdf(gamma_train_clf)
    assert subarray == SubarrayDescription.from_hdf(output), "Subarrays do not match"

    tel_groups = [
        "/dl1/event/telescope/parameters",
        "/dl1/event/telescope/images",
        "/dl2/event/telescope/impact/HillasReconstructor",
        "/dl2/event/telescope/energy/ExtraTreesRegressor",
        "/simulation/event/telescope/parameters",
        "/simulation/event/telescope/images",
        "/simulation/event/telescope/impact",
    ]

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

    in1 = tables.open_file(gamma_train_clf)
    in2 = tables.open_file(proton_train_clf)
    merged = tables.open_file(output)
    with in1, in2, merged:
        tables_checked = 0
        for table in tables_to_check:
            tables_checked += compare_table(in1, in2, merged, table)

        # regression test, no special meaning of the 83
        assert tables_checked == 83

        for table in statistics_tables:
            compare_stats_table(in1, in2, merged, table)

    product = metadata.read_ctao_metadata(output)
    assert product.data.level is dp.DataLevel.DL2


def test_append_requires_existing_output(tmp_path):
    from ctapipe.io.hdf5merger import HDF5Merger

    with pytest.raises(TraitError, match="file does not exist"):
        HDF5Merger(output_path=tmp_path / "missing.h5", append=True)


def test_append(tmp_path, gamma_train_clf, proton_train_clf):
    from ctapipe.io.hdf5merger import CannotMerge, HDF5Merger

    gamma_train_en = get_dataset_path("gamma_diffuse_dl2_train_small.dl2.h5")

    output = tmp_path / "merged_simple.dl2.h5"
    shutil.copy2(gamma_train_clf, output)

    with HDF5Merger(output, append=True) as merger:
        # this should work
        merger(proton_train_clf)

        # this shouldn't, because train_en does not already contain energy
        with pytest.raises(
            CannotMerge, match="Required node .*/energy/ExtraTreesRegressor"
        ):
            merger(gamma_train_en)


def test_contact_override(tmp_path, dl1_file):
    from ctapipe.io.hdf5merger import HDF5Merger

    output = tmp_path / "contact.h5"
    config = Config(
        {
            "HDF5Merger": {
                "Contact": {
                    "name": "Merge Contact",
                    "organization": "CTAO",
                    "email": "merge@example.org",
                }
            }
        }
    )

    with HDF5Merger(output_path=output, config=config) as merger:
        merger(dl1_file)

    product = metadata.read_ctao_metadata(output)
    assert product.contact == dp.Contact(
        name="Merge Contact",
        organization="CTAO",
        email="merge@example.org",
    )


@pytest.mark.parametrize(
    ("merge_strategy", "first_type", "second_type", "compatible"),
    [
        ("events-multiple-obs", dp.DataType.OBSERVATION, dp.DataType.OBSERVATION, True),
        (
            "events-multiple-obs",
            dp.DataType.OBSERVATION,
            dp.DataType.CALIBRATION,
            False,
        ),
        ("monitoring-only", dp.DataType.OBSERVATION, dp.DataType.CALIBRATION, True),
        (
            "monitoring-only",
            dp.DataType.OBSERVATION_SIM,
            dp.DataType.CALIBRATION_SIM,
            True,
        ),
        (
            "monitoring-only",
            dp.DataType.OBSERVATION,
            dp.DataType.OBSERVATION_SIM,
            False,
        ),
    ],
)
def test_data_type_compatibility(
    tmp_path, merge_strategy, first_type, second_type, compatible
):
    from ctapipe.io.hdf5merger import CannotMerge, HDF5Merger

    other = SimpleNamespace(filename="input.h5")
    with HDF5Merger(tmp_path / "merged.h5", merge_strategy=merge_strategy) as merger:
        merger.data_type = first_type

        if compatible:
            merger._check_data_type(other, second_type)
        else:
            with pytest.raises(CannotMerge, match="incompatible data type"):
                merger._check_data_type(other, second_type)


@pytest.mark.parametrize(
    ("merge_strategy", "second_version", "compatible"),
    [
        ("events-multiple-obs", "v7.6.0", True),
        ("events-multiple-obs", "v7.5.0", False),
        ("monitoring-only", "v7.5.0", True),
        ("monitoring-only", "v7.7.0", False),
    ],
)
def test_data_model_version_compatibility(
    tmp_path, merge_strategy, second_version, compatible
):
    from ctapipe.io.hdf5merger import CannotMerge, HDF5Merger

    other = SimpleNamespace(filename="input.h5")
    with HDF5Merger(tmp_path / "merged.h5", merge_strategy=merge_strategy) as merger:
        merger.data_model_version = "v7.6.0"

        if compatible:
            merger._check_data_model_version(other, second_version)
        else:
            with pytest.raises(CannotMerge, match="data model version"):
                merger._check_data_model_version(other, second_version)


def test_data_category_compatibility(tmp_path):
    from ctapipe.io.hdf5merger import CannotMerge, HDF5Merger

    other = SimpleNamespace(filename="input.h5")
    other_metadata = SimpleNamespace(
        model=SimpleNamespace(version="v7.6.0"),
        data=SimpleNamespace(type=dp.DataType.OBSERVATION_SIM),
        instance=SimpleNamespace(category=dp.DataProcessingCategory.A),
    )

    with HDF5Merger(tmp_path / "merged.h5") as merger:
        merger.data_model_version = "v7.6.0"
        merger.data_type = dp.DataType.OBSERVATION_SIM
        merger.data_category = dp.DataProcessingCategory.B
        merger.required_nodes = set()

        with pytest.raises(CannotMerge, match="different data category"):
            merger._check_can_merge(other, other_metadata)


def test_filter_column(tmp_path, dl2_shower_geometry_file):
    from ctapipe.io.hdf5merger import HDF5Merger

    output_path = tmp_path / "no_images.h5"
    with HDF5Merger(output_path, dl1_images=False, true_images=False) as merger:
        merger(dl2_shower_geometry_file)

    key = "/simulation/event/telescope/images/tel_003"
    with tables.open_file(output_path, "r") as f:
        assert key in f.root
        table = f.root[key]
        assert "true_image" not in table.colnames
        assert "CTAPIPE_VERSION" in table.attrs

    table = read_table(output_path, key)
    assert table["obs_id"].description == "Observation Block ID"


def test_muon(tmp_path, dl1_muon_output_file):
    from ctapipe.io.hdf5merger import HDF5Merger

    output = tmp_path / "muon_merged.dl2.h5"

    with HDF5Merger(output) as merger:
        merger(dl1_muon_output_file)

    table = read_table(output, "/dl1/event/telescope/muon/tel_001")
    input_table = read_table(dl1_muon_output_file, "/dl1/event/telescope/muon/tel_001")

    n_input = len(input_table)
    assert len(table) == n_input
    assert_table_equal(table, input_table)

    product = metadata.read_ctao_metadata(output)
    assert product.data.level is dp.DataLevel.DL1


def test_duplicated_obs_ids(tmp_path, dl2_shower_geometry_file):
    from ctapipe.io.hdf5merger import CannotMerge, HDF5Merger

    output = tmp_path / "invalid.dl1.h5"

    # check for fresh file
    with HDF5Merger(output) as merger:
        merger(dl2_shower_geometry_file)

        with pytest.raises(
            CannotMerge, match="Input file .* contains obs_ids already included"
        ):
            merger(dl2_shower_geometry_file)

    # check for appending
    with HDF5Merger(output, overwrite=True) as merger:
        merger(dl2_shower_geometry_file)

    with HDF5Merger(output, append=True) as merger:
        with pytest.raises(
            CannotMerge, match="Input file .* contains obs_ids already included"
        ):
            merger(dl2_shower_geometry_file)


def test_monitoring_traiterror(tmp_path, calibpipe_camcalib_sims_single_chunk):
    from ctapipe.core import traits
    from ctapipe.io.hdf5merger import HDF5Merger

    # Test if invalid monitoring settings raise TraitError
    with pytest.raises(
        traits.TraitError,
        match="Merge strategy 'monitoring-only' requires monitoring=True",
    ):
        _ = HDF5Merger(
            calibpipe_camcalib_sims_single_chunk,
            monitoring=False,
            merge_strategy="monitoring-only",
        )
