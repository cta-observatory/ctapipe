import shutil
import pytest
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
        ],
        cwd=cwd,
    )
    assert ret == 0, "Running stage1 failed"

    return output_path


@pytest.fixture
def gamma_dl1_path(tmp_path, prod5_gamma_simtel_path):
    dl1_file = tmp_path / "gamma.dl1.h5"
    return run_stage1(prod5_gamma_simtel_path, tmp_path, dl1_file)


@pytest.fixture
def proton_dl1_path(tmp_path, prod5_proton_simtel_path):
    dl1_file = tmp_path / "proton.dl1.h5"
    return run_stage1(prod5_proton_simtel_path, tmp_path, dl1_file)


def test_simple(tmp_path, gamma_dl1_path, proton_dl1_path):
    from ctapipe.tools.dl1_merge import MergeTool

    output = tmp_path / "merged_simple.dl1.h5"
    ret = run_tool(
        MergeTool(),
        argv=[
            str(gamma_dl1_path),
            str(proton_dl1_path),
            f"--output={output}",
            "--overwrite",
        ],
        cwd=tmp_path,
    )
    assert ret == 0
    run_stage1(output, cwd=tmp_path)


def test_pattern(tmp_path: Path, gamma_dl1_path, proton_dl1_path):
    from ctapipe.tools.dl1_merge import MergeTool

    # touch a random file to test that the pattern does not use it
    open(tmp_path / "foo.h5", "w").close()

    output = tmp_path / "merged_pattern.dl1.h5"
    ret = run_tool(
        MergeTool(),
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


def test_skip_images(tmp_path, gamma_dl1_path, proton_dl1_path):
    from ctapipe.tools.dl1_merge import MergeTool

    # create a second file so we can test the patterns
    output = tmp_path / "merged_no_images.dl1.h5"

    ret = run_tool(
        MergeTool(),
        argv=[
            str(gamma_dl1_path),
            str(proton_dl1_path),
            f"--output={output}",
            "--skip-images",
            "--overwrite",
        ],
        cwd=tmp_path,
    )
    assert ret == 0
