#!/usr/bin/env python3

"""
Test ctapipe-process on a few different use cases
"""

import numpy as np
import pandas as pd
import pytest
import tables

from ctapipe.core import run_tool
from ctapipe.instrument.subarray import SubarrayDescription
from ctapipe.io import DataLevel, EventSource, read_table
from ctapipe.tools.process import ProcessorTool
from ctapipe.tools.quickstart import CONFIGS_TO_WRITE, QuickStartTool
from ctapipe.utils import get_dataset_path

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

GAMMA_TEST_LARGE = get_dataset_path("gamma_test_large.simtel.gz")


def resource_file(filename):
    return files("ctapipe").joinpath("resources", filename)


@pytest.mark.parametrize(
    "config_files",
    [
        ("base_config.yaml", "stage1_config.yaml"),
        ("stage1_config.toml",),
        ("stage1_config.json",),
    ],
)
def test_read_yaml_toml_json_config(dl1_image_file, config_files):
    """check that we can read multiple formats of config file"""
    tool = ProcessorTool()

    for config_base in config_files:
        config = resource_file(config_base)
        tool.load_config_file(config)

    tool.config.EventSource.input_url = dl1_image_file
    tool.config.DataWriter.overwrite = True
    tool.setup()
    assert (
        tool.get_current_config()["ProcessorTool"]["DataWriter"]["contact_info"].name
        == "YOUR-NAME-HERE"
    )


def test_multiple_configs(dl1_image_file):
    """ensure a config file loaded later overwrites keys from an earlier one"""
    tool = ProcessorTool()

    tool.load_config_file(resource_file("base_config.yaml"))
    tool.load_config_file(resource_file("stage2_config.yaml"))

    tool.config.EventSource.input_url = dl1_image_file
    tool.config.DataWriter.overwrite = True
    tool.setup()

    # ensure the overwriting works (base config has this option disabled)
    assert (
        tool.get_current_config()["ProcessorTool"]["DataWriter"]["write_showers"]
        is True
    )


def test_stage_1_dl1(tmp_path, dl1_image_file, dl1_parameters_file):
    """check simtel to DL1 conversion"""
    config = resource_file("stage1_config.json")

    # DL1A file as input
    dl1b_from_dl1a_file = tmp_path / "dl1b_fromdl1a.dl1.h5"
    assert (
        run_tool(
            ProcessorTool(),
            argv=[
                f"--config={config}",
                f"--input={dl1_image_file}",
                f"--output={dl1b_from_dl1a_file}",
                "--camera-frame",
                "--write-parameters",
                "--overwrite",
            ],
            cwd=tmp_path,
        )
        == 0
    )

    # check tables were written
    with tables.open_file(dl1b_from_dl1a_file, mode="r") as testfile:
        assert testfile.root.dl1
        assert testfile.root.dl1.event.telescope
        assert testfile.root.dl1.event.subarray
        assert testfile.root.configuration.instrument.subarray.layout
        assert testfile.root.configuration.instrument.telescope.optics
        assert testfile.root.configuration.instrument.telescope.camera.geometry_0
        assert testfile.root.configuration.instrument.telescope.camera.readout_0

        assert testfile.root.dl1.monitoring.subarray.pointing.dtype.names == (
            "time",
            "array_azimuth",
            "array_altitude",
            "array_ra",
            "array_dec",
        )

    # check we can read telescope parameters
    dl1_features = pd.read_hdf(
        dl1b_from_dl1a_file, "/dl1/event/telescope/parameters/tel_025"
    )
    features = (
        "obs_id",
        "event_id",
        "tel_id",
        "camera_frame_hillas_intensity",
        "camera_frame_hillas_x",
        "concentration_cog",
        "leakage_pixels_width_1",
    )
    for feature in features:
        assert feature in dl1_features.columns

    true_impact = read_table(
        dl1b_from_dl1a_file,
        "/simulation/event/telescope/impact/tel_025",
    )
    assert "true_impact_distance" in true_impact.colnames

    # DL1B file as input
    ret = run_tool(
        ProcessorTool(),
        argv=[
            f"--config={config}",
            f"--input={dl1_parameters_file}",
            f"--output={tmp_path}/dl1b_from_dl1b.dl1.h5",
            "--write-parameters",
            "--overwrite",
        ],
        cwd=tmp_path,
    )
    assert ret == 1


def test_stage1_datalevels(tmp_path):
    """test the dl1 tool on a file not providing r1, dl0 or dl1a"""

    class DummyEventSource(EventSource):
        """for testing"""

        @staticmethod
        def is_compatible(file_path):
            with open(file_path, "rb") as infile:
                dummy = infile.read(5)
                return dummy == b"dummy"

        @property
        def datalevels(self):
            return (DataLevel.R0,)

        @property
        def is_simulation(self):
            return True

        @property
        def scheduling_blocks(self):
            return dict()

        @property
        def observation_blocks(self):
            return dict()

        @property
        def subarray(self):
            return None

        def _generator(self):
            return None

    dummy_file = tmp_path / "datalevels_dummy.h5"
    out_file = tmp_path / "datalevels_dummy_stage1_output.h5"
    with open(dummy_file, "wb") as infile:
        infile.write(b"dummy")
        infile.flush()

    config = resource_file("stage1_config.json")
    tool = ProcessorTool()

    assert (
        run_tool(
            tool,
            argv=[
                f"--config={config}",
                f"--input={dummy_file}",
                f"--output={out_file}",
                "--write-images",
                "--overwrite",
            ],
            cwd=tmp_path,
        )
        == 1
    )
    # make sure the dummy event source was really used
    assert isinstance(tool.event_source, DummyEventSource)


def test_stage_2_from_simtel(tmp_path):
    """check we can go to DL2 geometry from simtel file"""
    config = resource_file("stage2_config.json")
    output = tmp_path / "test_stage2_from_simtel.DL2.h5"

    assert (
        run_tool(
            ProcessorTool(),
            argv=[
                f"--config={config}",
                "--input=dataset://gamma_prod5.simtel.zst",
                f"--output={output}",
                "--overwrite",
            ],
            cwd=tmp_path,
        )
        == 0
    )

    # check tables were written
    with tables.open_file(output, mode="r") as testfile:
        dl2 = read_table(
            testfile,
            "/dl2/event/subarray/geometry/HillasReconstructor",
        )
        subarray = SubarrayDescription.from_hdf(testfile)

        # test tel_ids are included and transformed correctly
        assert "HillasReconstructor_telescopes" in dl2.colnames
        assert dl2["HillasReconstructor_telescopes"].dtype == np.bool_
        assert dl2["HillasReconstructor_telescopes"].shape[1] == len(subarray)


def test_stage_2_from_dl1_images(tmp_path, dl1_image_file):
    """check we can go to DL2 geometry from DL1 images"""
    config = resource_file("stage2_config.json")
    output = tmp_path / "test_stage2_from_dl1image.DL2.h5"

    assert (
        run_tool(
            ProcessorTool(),
            argv=[
                f"--config={config}",
                f"--input={dl1_image_file}",
                f"--output={output}",
                "--overwrite",
            ],
            cwd=tmp_path,
        )
        == 0
    )

    # check tables were written
    with tables.open_file(output, mode="r") as testfile:
        assert testfile.root.dl2.event.subarray.geometry.HillasReconstructor


def test_stage_2_from_dl1_params(tmp_path, dl1_parameters_file):
    """check we can go to DL2 geometry from DL1 parameters"""

    config = resource_file("stage2_config.json")
    output = tmp_path / "test_stage2_from_dl1param.DL2.h5"

    assert (
        run_tool(
            ProcessorTool(),
            argv=[
                f"--config={config}",
                f"--input={dl1_parameters_file}",
                f"--output={output}",
                "--overwrite",
            ],
            cwd=tmp_path,
        )
        == 0
    )

    # check tables were written
    with tables.open_file(output, mode="r") as testfile:
        assert testfile.root.dl2.event.subarray.geometry.HillasReconstructor


def test_training_from_simtel(tmp_path):
    """check we can write both dl1 and dl2 info (e.g. for training input)"""

    config = resource_file("training_config.json")
    output = tmp_path / "test_training.DL1DL2.h5"

    assert (
        run_tool(
            ProcessorTool(),
            argv=[
                f"--config={config}",
                f"--input={GAMMA_TEST_LARGE}",
                f"--output={output}",
                "--max-events=5",
                "--overwrite",
                "--SimTelEventSource.focal_length_choice=EQUIVALENT",
            ],
            cwd=tmp_path,
        )
        == 0
    )

    # check tables were written
    with tables.open_file(output, mode="r") as testfile:
        assert testfile.root.dl1.event.telescope.parameters.tel_002
        assert testfile.root.dl2.event.subarray.geometry.HillasReconstructor


def test_image_modifications(tmp_path, dl1_image_file):
    """
    Test that running ctapipe-process with an ImageModifier set
    produces a file with different images.
    """

    unmodified_images = read_table(
        dl1_image_file, "/dl1/event/telescope/images/tel_025"
    )
    noise_config = resource_file("image_modification_config.json")

    dl1_modified = tmp_path / "dl1_modified.dl1.h5"
    assert (
        run_tool(
            ProcessorTool(),
            argv=[
                f"--config={noise_config}",
                f"--input={dl1_image_file}",
                f"--output={dl1_modified}",
                "--write-parameters",
                "--overwrite",
            ],
            cwd=tmp_path,
        )
        == 0
    )
    modified_images = read_table(dl1_modified, "/dl1/event/telescope/images/tel_025")
    # Test that significantly more light is recorded (bias in dim pixels)
    assert modified_images["image"].sum() / unmodified_images["image"].sum() > 1.5


@pytest.mark.parametrize(
    "filename", ["base_config.yaml", "stage1_config.json", "stage1_config.toml"]
)
def test_quickstart_templates(filename):
    """ensure template configs have an appropriate placeholder for the contact info"""
    config = resource_file(filename)
    text = config.read_text()

    assert "YOUR-NAME-HERE" in text, "Missing expected name placeholder"
    assert "YOUREMAIL@EXAMPLE.ORG" in text, "Missing expected email placeholder"
    assert "YOUR-ORGANIZATION" in text, "Missing expected org placeholder"


def test_quickstart(tmp_path):
    """ensure quickstart tool generates expected output"""

    tool = QuickStartTool()
    run_tool(
        tool,
        cwd=tmp_path,
        argv=[
            "--workdir",
            "ProdX",
            "--name",
            "test",
            "--email",
            "a@b.com",
            "--org",
            "CTA",
        ],
    )

    assert (tmp_path / "ProdX" / "README.md").exists()

    for config in CONFIGS_TO_WRITE:
        assert (tmp_path / "ProdX" / config).exists()
