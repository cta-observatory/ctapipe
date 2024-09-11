import shutil
import tempfile
from importlib.resources import files
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
import tables
from astropy.table import vstack
from astropy.utils.diff import report_diff_values

from ctapipe.core import ToolConfigurationError, run_tool
from ctapipe.io import TableLoader
from ctapipe.io.astropy_helpers import read_table
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
    assert (
        identical
    ), f"Merged table not equal to individual tables. Diff:\n {diff.getvalue()}"

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
