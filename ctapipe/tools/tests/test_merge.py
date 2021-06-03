import tables
import tempfile

from ctapipe.core import run_tool
from pathlib import Path

from ctapipe.tools.stage1 import Stage1Tool


def run_stage1(input_path, cwd, output_path=None):
    config = Path("./examples/stage1_config.json").absolute()

    if output_path is None:
        output_path = Path(
            tempfile.NamedTemporaryFile(suffix=".dl1.h5", dir=cwd).name
        ).absolute()

    ret = run_tool(
        Stage1Tool(),
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
    from ctapipe.tools.dl1_merge import MergeTool

    output = tmp_path / "merged_simple.dl1.h5"
    ret = run_tool(
        MergeTool(),
        argv=[str(dl1_file), str(dl1_proton_file), f"--output={output}", "--overwrite"],
        cwd=tmp_path,
    )
    assert ret == 0
    run_stage1(output, cwd=tmp_path)


def test_pattern(tmp_path: Path, dl1_file, dl1_proton_file):
    from ctapipe.tools.dl1_merge import MergeTool

    # touch a random file to test that the pattern does not use it
    open(dl1_file.parent / "foo.h5", "w").close()

    output = tmp_path / "merged_pattern.dl1.h5"
    ret = run_tool(
        MergeTool(),
        argv=[
            "-i",
            str(dl1_file.parent),
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
    from ctapipe.tools.dl1_merge import MergeTool

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
    from ctapipe.tools.dl1_merge import MergeTool
    from ctapipe.instrument import SubarrayDescription

    # create file to test 'allowed-tels' option
    output = tmp_path / "merged_allowed_tels.dl1.h5"
    ret = run_tool(
        MergeTool(),
        argv=[
            str(dl1_file),
            str(dl1_proton_file),
            f"--output={output}",
            "--allowed-tels=[1,2]",
            "--overwrite",
        ],
        cwd=tmp_path,
    )
    assert ret == 0

    s = SubarrayDescription.from_hdf(output)
    assert s.tel.keys() == {1, 2}
