import tables
import tempfile
import shutil

from ctapipe.core import run_tool
from pathlib import Path
from ctapipe.io.astropy_helpers import read_table
from astropy.table import vstack
from astropy.utils.diff import report_diff_values
from io import StringIO

from ctapipe.tools.process import ProcessorTool

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files


def run_stage1(input_path, cwd, output_path=None):
    config = files("ctapipe.tools.tests").joinpath("resources", "stage1_config.json")

    if output_path is None:
        output_path = Path(
            tempfile.NamedTemporaryFile(suffix=".dl1.h5", dir=cwd).name
        ).absolute()

    ret = run_tool(
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
    )
    assert ret == 0, "Running stage1 failed"


def test_simple(tmp_path, dl1_file, dl1_proton_file):
    from ctapipe.tools.merge import MergeTool

    output = tmp_path / "merged_simple.dl1.h5"
    ret = run_tool(
        MergeTool(),
        argv=[str(dl1_file), str(dl1_proton_file), f"--output={output}", "--overwrite"],
        cwd=tmp_path,
    )
    assert ret == 0
    run_stage1(output, cwd=tmp_path)


def test_pattern(tmp_path: Path, dl1_file, dl1_proton_file):
    from ctapipe.tools.merge import MergeTool

    # touch a random file to test that the pattern does not use it
    open(dl1_file.parent / "foo.h5", "w").close()

    # copy to make sure we don't have other files in the dl1 dir disturb this
    for f in (dl1_file, dl1_proton_file):
        shutil.copy(f, tmp_path)

    output = tmp_path / "merged_pattern.dl1.h5"
    ret = run_tool(
        tool=MergeTool(),
        argv=[
            "-i",
            str(tmp_path),
            "-p",
            "*.dl1.h5",
            f"--output={output}",
            "--overwrite",
        ],
        cwd=tmp_path,
    )
    assert ret == 0
    run_stage1(output, cwd=tmp_path)


def test_skip_images(tmp_path, dl1_file, dl1_proton_file):
    from ctapipe.tools.merge import MergeTool

    # create a second file so we can test the patterns
    output = tmp_path / "merged_no_images.dl1.h5"
    ret = run_tool(
        MergeTool(),
        argv=[
            str(dl1_file),
            str(dl1_proton_file),
            f"--output={output}",
            "--skip-images",
            "--overwrite",
        ],
        cwd=tmp_path,
    )

    with tables.open_file(output, "r") as f:
        assert "images" not in f.root.dl1.event.telescope
        assert "parameters" in f.root.dl1.event.telescope

    assert ret == 0


def test_allowed_tels(tmp_path, dl1_file, dl1_proton_file):
    from ctapipe.tools.merge import MergeTool
    from ctapipe.instrument import SubarrayDescription

    # create file to test 'allowed-tels' option
    output = tmp_path / "merged_allowed_tels.dl1.h5"

    allowed_tels = {25, 125}

    argv = [str(dl1_file), str(dl1_proton_file), f"--output={output}", "--overwrite"]
    for tel_id in allowed_tels:
        argv.append(f"--allowed-tels={tel_id}")

    ret = run_tool(MergeTool(), argv=argv, cwd=tmp_path)
    assert ret == 0

    s = SubarrayDescription.from_hdf(output)
    assert s.tel.keys() == allowed_tels

    tel_keys = {f"tel_{tel_id:03d}" for tel_id in allowed_tels}
    with tables.open_file(output) as f:
        assert set(f.root.dl1.event.telescope.parameters._v_children).issubset(tel_keys)
        assert set(f.root.dl1.event.telescope.images._v_children).issubset(tel_keys)
        assert set(f.root.dl1.monitoring.telescope.pointing._v_children).issubset(
            tel_keys
        )


def test_dl2(tmp_path, dl2_shower_geometry_file, dl2_proton_geometry_file):
    from ctapipe.tools.merge import MergeTool

    output = tmp_path / "merged.dl2.h5"
    ret = run_tool(
        MergeTool(),
        argv=[
            f"--output={output}",
            str(dl2_shower_geometry_file),
            str(dl2_proton_geometry_file),
        ],
    )
    assert ret == 0, f"Running merge for dl2 files failed with exit code {ret}"

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
