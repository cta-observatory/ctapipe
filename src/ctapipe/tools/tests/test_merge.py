import shutil
import tempfile
from contextlib import ExitStack
from importlib.resources import files
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
import tables
from astropy.table import vstack
from astropy.utils.diff import report_diff_values
from ctao_datamodel.models import dataproducts as dp
from ctao_datamodel.models.common import SiteID

from ctapipe.core import ToolConfigurationError, run_tool
from ctapipe.io import DataWriter, EventSource, TableLoader
from ctapipe.io import metadata as meta
from ctapipe.io.astropy_helpers import read_table
from ctapipe.io.datalevels import DataLevel
from ctapipe.io.tests.test_astropy_helpers import assert_table_equal
from ctapipe.tools.process import ProcessorTool


def run_stage1(input_path, cwd, output_path=None):
    config = files("ctapipe").joinpath("resources", "stage1_config.json")

    if output_path is None:
        output_path = Path(
            tempfile.NamedTemporaryFile(suffix=".dl1.h5", dir=cwd).name
        ).absolute()

    run_tool(
        ProcessorTool(),
        argv=[
            f"--config={config}",
            f"--input={input_path}",
            f"--output={output_path}",
            "--write-parameters",
            "--write-images",
            "--overwrite",
            "--max-events=5",
        ],
        cwd=cwd,
        raises=True,
    )


def test_simple(tmp_path, dl1_file, dl1_proton_file):
    from ctapipe.tools.merge import MergeTool

    output = tmp_path / "merged_simple.dl1.h5"
    run_tool(
        MergeTool(),
        argv=[str(dl1_file), str(dl1_proton_file), f"--output={output}", "--overwrite"],
        cwd=tmp_path,
        raises=True,
    )
    run_stage1(output, cwd=tmp_path)


def test_migrates_legacy_metadata(tmp_path, dl1_file):
    """The tool must migrate headers, not just rely on the conversion helper."""
    from ctapipe.tools.merge import MergeTool

    legacy_input = tmp_path / "legacy.dl1.h5"
    output = tmp_path / "migrated.dl1.h5"
    shutil.copy2(dl1_file, legacy_input)

    legacy = meta.Reference(
        contact=meta.Contact(
            name="Legacy Contact",
            organization="Legacy Organization",
            email="legacy@example.org",
        ),
        product=meta.Product(
            description="Legacy DL1 images",
            creation_time="2020-10-11 15:23:31",
            data_category="B",
            data_levels=[DataLevel.DL1_IMAGES],
            data_association="Subarray",
            data_model_name="ctapipe",
            data_model_version="v7.6.0",
            data_model_url="https://example.org/legacy-model",
            format="hdf5",
        ),
        process=meta.Process(type_="Simulation", subtype="legacy", id_="42"),
        activity=meta.Activity(
            name="legacy-process",
            id_="f93617ec-ef95-4147-b627-01f81e8cf2b4",
            start_time="2020-10-11 15:00:00",
            stop_time="2020-10-11 15:20:00",
            software_name="legacy-ctapipe",
            software_version="0.1",
        ),
        instrument=meta.Instrument(
            site="South",
            class_="Subarray",
            type_="legacy-array",
            subtype="legacy-subtype",
            version="legacy-version",
            id_="17",
        ),
    )

    with tables.open_file(legacy_input, mode="a") as h5file:
        attributes = h5file.root._v_attrs
        for name in attributes._f_list("user"):
            if name.startswith("CTAO.") or name.startswith("CTA "):
                del attributes[name]
        meta.write_to_hdf5(legacy.to_dict(), h5file)

    with pytest.warns(meta.LegacyMetadataWarning, match="Legacy"):
        run_tool(
            MergeTool(),
            argv=[str(legacy_input), f"--output={output}"],
            cwd=tmp_path,
            raises=True,
        )

    product = meta.read_ctao_metadata(output)
    assert product.description == "Legacy DL1 images"
    assert product.data == dp.ProductType(
        level=dp.DataLevel.DL1,
        division=dp.DataDivision.EVENT,
        association=dp.DataAssociation.SUBARRAY,
        type=dp.DataType.OBSERVATION_SIM,
    )
    assert product.instance.sublevel_id is dp.ProcessingSublevel.IMAGES
    assert product.instance.category is dp.DataProcessingCategory.B
    assert product.instance.site_id is SiteID.CTAO_SOUTH
    assert product.instance.subarray_id == 17
    assert product.contact == dp.Contact(
        name="Legacy Contact",
        organization="Legacy Organization",
        email="legacy@example.org",
    )
    assert product.model.name == "ctapipe"
    assert product.model.version == "v7.6.0"
    assert str(product.model.url) == "https://example.org/legacy-model"
    assert product.activity.name == "ctapipe-merge"

    with tables.open_file(output) as h5file:
        names = h5file.root._v_attrs._f_list("user")
        assert "CTAO.ctao_metadata_version" in names
        assert not any(name.startswith("CTA ") for name in names)


def test_migrates_legacy_dataset_metadata(
    tmp_path, calibpipe_camcalib_sims_single_chunk
):
    """Test migration using an unchanged legacy file from the test dataset."""
    from ctapipe.tools.merge import MergeTool

    output = tmp_path / "migrated_dataset.dl1.h5"

    with pytest.warns(meta.LegacyMetadataWarning):
        run_tool(
            MergeTool(),
            argv=[
                str(calibpipe_camcalib_sims_single_chunk),
                f"--output={output}",
            ],
            cwd=tmp_path,
            raises=True,
        )

    product = meta.read_ctao_metadata(output)
    assert product.description == "ctapipe Data Product"
    assert product.data == dp.ProductType(
        level=dp.DataLevel.DL1,
        division=dp.DataDivision.EVENT,
        association=dp.DataAssociation.SUBARRAY,
        type=dp.DataType.OBSERVATION_SIM,
    )
    assert product.instance.sublevel_id is dp.ProcessingSublevel.IMAGES
    assert product.model.name == "ASWG"
    assert product.model.version == "v7.2.0"
    assert product.contact.email == "unknown@example.org"
    assert product.activity.name == "ctapipe-merge"

    with tables.open_file(output) as h5file:
        names = h5file.root._v_attrs._f_list("user")
        assert "CTAO.ctao_metadata_version" in names
        assert not any(name.startswith("CTA ") for name in names)


def test_monitoring_only_append_updates_data_type(
    tmp_path,
    dl1_tel1_file,
    calibpipe_camcalib_sims_single_chunk,
):
    from ctapipe.tools.merge import MergeTool

    output = tmp_path / "monitoring_only.dl1.h5"
    shutil.copy2(dl1_tel1_file, output)

    with pytest.warns(meta.LegacyMetadataWarning):
        run_tool(
            MergeTool(),
            argv=[
                str(calibpipe_camcalib_sims_single_chunk),
                f"--output={output}",
                "--append",
                "--merge-strategy=monitoring-only",
            ],
            cwd=tmp_path,
            raises=True,
        )

    product = meta.read_ctao_metadata(output)
    assert product.data.division is dp.DataDivision.MONITORING
    assert product.data.type is dp.DataType.CALIBRATION_SIM


def test_pattern(tmp_path: Path, dl1_file, dl1_proton_file):
    from ctapipe.tools.merge import MergeTool

    # touch a random file to test that the pattern does not use it
    open(dl1_file.parent / "foo.h5", "w").close()

    # copy to make sure we don't have other files in the dl1 dir disturb this
    indir = tmp_path / "input"
    indir.mkdir()
    for f in (dl1_file, dl1_proton_file):
        shutil.copy(f, indir)

    output = tmp_path / "merged_pattern.dl1.h5"
    run_tool(
        tool=MergeTool(),
        argv=[
            "-i",
            str(indir),
            "-p",
            "*.dl1.h5",
            f"--output={output}",
            "--overwrite",
        ],
        cwd=tmp_path,
        raises=True,
    )
    run_stage1(output, cwd=tmp_path)


def test_skip_images(tmp_path, dl1_file, dl1_proton_file):
    from ctapipe.tools.merge import MergeTool

    # create a second file so we can test the patterns
    output = tmp_path / "merged_no_images.dl1.h5"
    run_tool(
        MergeTool(),
        argv=[
            str(dl1_file),
            str(dl1_proton_file),
            f"--output={output}",
            "--no-dl1-images",
            "--no-true-images",
            "--overwrite",
        ],
        cwd=tmp_path,
        raises=True,
    )

    with tables.open_file(output, "r") as f:
        assert "images" not in f.root.dl1.event.telescope
        assert "images" in f.root.simulation.event.telescope
        assert "parameters" in f.root.dl1.event.telescope

    t = read_table(output, "/simulation/event/telescope/images/tel_001")
    assert "true_image" not in t.colnames
    assert "true_image_sum" in t.colnames


def test_dl2(tmp_path, dl2_shower_geometry_file, dl2_proton_geometry_file):
    from ctapipe.tools.merge import MergeTool

    output = tmp_path / "merged.dl2.h5"
    run_tool(
        MergeTool(),
        argv=[
            f"--output={output}",
            str(dl2_shower_geometry_file),
            str(dl2_proton_geometry_file),
        ],
        raises=True,
    )

    table1 = read_table(
        dl2_shower_geometry_file, "/dl2/event/subarray/geometry/HillasReconstructor"
    )
    table2 = read_table(
        dl2_proton_geometry_file, "/dl2/event/subarray/geometry/HillasReconstructor"
    )
    table_merged = read_table(
        output, "/dl2/event/subarray/geometry/HillasReconstructor"
    )

    diff = StringIO()
    identical = report_diff_values(vstack([table1, table2]), table_merged, fileobj=diff)
    assert identical, (
        f"Merged table not equal to individual tables. Diff:\n {diff.getvalue()}"
    )

    stats_key = "/dl2/service/tel_event_statistics/HillasReconstructor"
    merged_stats = read_table(output, stats_key)
    stats1 = read_table(dl2_shower_geometry_file, stats_key)
    stats2 = read_table(dl2_proton_geometry_file, stats_key)

    for col in ("counts", "cumulative_counts"):
        assert np.all(merged_stats[col] == (stats1[col] + stats2[col]))

    # test reading configurations as well:
    obs = read_table(output, "/configuration/observation/observation_block")
    sbs = read_table(output, "/configuration/observation/scheduling_block")

    assert len(obs) == 2, "should have two OB entries"
    assert len(sbs) == 2, "should have two SB entries"

    # regression test for #2048
    loader = TableLoader(output)
    tel_events = loader.read_telescope_events(
        dl1_parameters=False,
        true_parameters=False,
    )
    assert "true_impact_distance" in tel_events.colnames
    # regression test for #2051
    assert "HillasReconstructor_tel_impact_distance" in tel_events.colnames


def test_muon(tmp_path, dl1_muon_output_file):
    from ctapipe.tools.merge import MergeTool

    output = tmp_path / "muon_merged.dl2.h5"
    run_tool(
        MergeTool(),
        argv=[
            f"--output={output}",
            str(dl1_muon_output_file),
        ],
        raises=True,
    )

    table = read_table(output, "/dl1/event/telescope/muon/tel_001")
    input_table = read_table(dl1_muon_output_file, "/dl1/event/telescope/muon/tel_001")

    n_input = len(input_table)
    assert len(table) == n_input
    assert_table_equal(table, input_table)


def test_duplicated(tmp_path, dl1_file, dl1_proton_file):
    from ctapipe.tools.merge import MergeTool

    output = tmp_path / "invalid.dl1.h5"
    with pytest.raises(ToolConfigurationError, match="Same file given multiple times"):
        run_tool(
            MergeTool(),
            argv=[
                str(dl1_file),
                str(dl1_proton_file),
                str(dl1_file),
                f"--output={output}",
                "--overwrite",
            ],
            cwd=tmp_path,
            raises=True,
        )


@pytest.fixture(scope="session")
def dl1_chunks(tmp_path_factory, dl1_file):
    outdir = tmp_path_factory.mktemp("dl1_chunks_")
    # write two chunks from the same simulation run, merged result should
    # match initial input
    path1 = outdir / "single_ob_1.dl1.h5"
    path2 = outdir / "single_ob_2.dl1.h5"
    ctx = ExitStack()
    with ctx:
        source = ctx.enter_context(EventSource(dl1_file))
        writer_kwargs = dict(event_source=source, write_dl1_images=True)
        writer1 = ctx.enter_context(DataWriter(output_path=path1, **writer_kwargs))
        writer2 = ctx.enter_context(DataWriter(output_path=path2, **writer_kwargs))

        for event in source:
            writer = writer1 if event.count < 3 else writer2
            writer(event)

    return path1, path2


def test_merge_single_ob(tmp_path, dl1_file, dl1_chunks):
    from ctapipe.tools.merge import MergeTool

    path1, path2 = dl1_chunks

    output = tmp_path / "single_ob.dl1.h5"
    run_tool(
        MergeTool(),
        argv=[
            str(path1),
            str(path2),
            f"--output={output}",
            "--single-ob",
        ],
        cwd=tmp_path,
        raises=True,
    )

    with TableLoader(output) as loader:
        merged_tel_events = loader.read_telescope_events()

    with TableLoader(dl1_file) as loader:
        initial_tel_events = loader.read_telescope_events()

    assert_table_equal(merged_tel_events, initial_tel_events)


def test_merge_single_ob_append(tmp_path, dl1_file, dl1_chunks):
    from ctapipe.tools.merge import MergeTool

    path1, path2 = dl1_chunks

    output = tmp_path / "single_ob.dl1.h5"
    run_tool(
        MergeTool(),
        argv=[
            str(path1),
            f"--output={output}",
            "--single-ob",
        ],
        cwd=tmp_path,
        raises=True,
    )

    run_tool(
        MergeTool(),
        argv=[
            str(path2),
            f"--output={output}",
            "--single-ob",
            "--append",
        ],
        cwd=tmp_path,
        raises=True,
    )

    with TableLoader(output) as loader:
        merged_tel_events = loader.read_telescope_events()

    with TableLoader(dl1_file) as loader:
        initial_tel_events = loader.read_telescope_events()

    assert_table_equal(merged_tel_events, initial_tel_events)


def test_merge_exceptions(
    tmp_path, calibpipe_camcalib_sims_single_chunk, dl1_mon_pointing_file
):
    from ctapipe.io.hdf5merger import CannotMerge
    from ctapipe.tools.merge import MergeTool

    # Test if invalid merge with different monitoring types raises CannotMerge
    with pytest.raises(CannotMerge, match="Required node"):
        argv = [
            f"--output={calibpipe_camcalib_sims_single_chunk}",
            str(dl1_mon_pointing_file),
            "--append",
            "--monitoring",
            "--single-ob",
        ]
        run_tool(MergeTool(), argv=argv, cwd=tmp_path)
