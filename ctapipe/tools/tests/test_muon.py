"""
Test individual muon tool functionality
"""

import subprocess
import pytest

import tempfile
import tables

from ctapipe.utils import get_dataset_path
from ctapipe.core import run_tool
import numpy as np

tmp_dir = tempfile.TemporaryDirectory()
LST_MUONS = get_dataset_path("lst_muons.simtel.zst")


@pytest.fixture(scope="module")
def dl1_muon_file():
    """
    DL1 file containing only images from a muon simulation set.
    """
    command = (
        "ctapipe-stage1 "
        f"--input {LST_MUONS} "
        f"--output {tmp_dir.name}/muons.dl1.h5 "
        "--write-images"
    )
    subprocess.call(command.split(), stdout=subprocess.PIPE)
    return f"{tmp_dir.name}/muons.dl1.h5"


def test_muon_reconstruction(tmpdir, dl1_muon_file):
    from ctapipe.tools.stage1 import Stage1Tool

    muon_simtel_output_file = tmp_dir.name + "/muon_reco_on_simtel.h5"
    assert (
        run_tool(
            Stage1Tool(),
            argv=[
                f"--input={LST_MUONS}",
                f"--output={muon_simtel_output_file}",
                "--overwrite",
                "--write-muons",
            ],
            cwd=tmpdir,
        )
        == 0
    )

    with tables.open_file(muon_simtel_output_file) as t:
        table = t.root.dl1.event.telescope.muon_parameters.tel_001[:]
        assert len(table) > 20
        assert np.count_nonzero(np.isnan(table["muonring_radius"])) == 0

    muon_dl1_output_file = tmp_dir.name + "/muon_reco_on_dl1a.h5"
    assert (
        run_tool(
            Stage1Tool(),
            argv=[
                f"--input={dl1_muon_file}",
                f"--output={muon_dl1_output_file}",
                "--overwrite",
                "--write-muons",
            ],
            cwd=tmpdir,
        )
        == 0
    )

    with tables.open_file(muon_dl1_output_file) as t:
        table = t.root.dl1.event.telescope.muon_parameters.tel_001[:]
        assert len(table) > 20
        assert np.count_nonzero(np.isnan(table["muonring_radius"])) == 0

    assert run_tool(Stage1Tool(), ["--help-all"]) == 0
