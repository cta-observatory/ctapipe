
import numpy as np
import pandas as pd
import pytest
import tables

from ctapipe.core import run_tool
from ctapipe.tools.process import ProcessorTool
from ctapipe.utils import get_dataset_path, resource_file
from ctapipe.io.hdf5dataformat import SIMULATION_PARAMETERS_GROUP

def test_true_disp_calculation(tmp_path, dl1_image_file):
    """check true_disp calculation in ctapipe-process"""
    print("DEBUG: Starting test_true_disp_calculation", flush=True)
    config = resource_file("stage1_config.json")
    
    output_file = tmp_path / "true_disp_test.dl1.h5"
    
    run_tool(
        ProcessorTool(),
        argv=[
            f"--config={config}",
            f"--input={dl1_image_file}",
            f"--output={output_file}",
            "--write-parameters",
            "--overwrite",
        ],
        cwd=tmp_path,
        raises=True,
    )

    # check if fields exist and are not all NaN
    # We need to find a telescope that has events
    
    with tables.open_file(output_file, mode="r") as testfile:
        # Check if true parameters group exists for at least one telescope
        sim_params_group = testfile.root.simulation.event.telescope.parameters
        assert len(sim_params_group._v_children) > 0
        
        first_tel_group = list(sim_params_group._v_children.keys())[0]
        
    true_params = pd.read_hdf(
        output_file, f"{SIMULATION_PARAMETERS_GROUP}/{first_tel_group}"
    )

    assert "true_disp" in true_params.columns
    
    # Check that we have some valid values
    assert np.count_nonzero(np.isfinite(true_params["true_disp"])) > 0
