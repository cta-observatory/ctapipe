#!/usr/bin/env python3

"""
Test ctapipe-process on a few different use cases
"""

from pathlib import Path

import pandas as pd
import tables
from ctapipe.core import run_tool
from ctapipe.tools.process import ProcessorTool
from ctapipe.utils import get_dataset_path

GAMMA_TEST_LARGE = get_dataset_path("gamma_test_large.simtel.gz")


def test_stage_1_dl1(tmp_path, dl1_image_file, dl1_parameters_file):
    from ctapipe.tools.process import ProcessorTool

    config = Path("./examples/stage1_config.json").absolute()
    # DL1A file as input
    dl1b_from_dl1a_file = tmp_path / "dl1b_fromdl1a.dl1.h5"
    assert (
        run_tool(
            ProcessorTool(),
            argv=[
                f"--config={config}",
                f"--input={dl1_image_file}",
                f"--output={dl1b_from_dl1a_file}",
                "--write-parameters",
                "--overwrite",
            ],
            cwd=tmp_path,
        )
        == 0
    )

    # check tables were written
    with tables.open_file(dl1b_from_dl1a_file, mode="r") as tf:
        assert tf.root.dl1
        assert tf.root.dl1.event.telescope
        assert tf.root.dl1.event.subarray
        assert tf.root.configuration.instrument.subarray.layout
        assert tf.root.configuration.instrument.telescope.optics
        assert tf.root.configuration.instrument.telescope.camera.geometry_LSTCam
        assert tf.root.configuration.instrument.telescope.camera.readout_LSTCam

        assert tf.root.dl1.monitoring.subarray.pointing.dtype.names == (
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
        "hillas_intensity",
        "concentration_cog",
        "leakage_pixels_width_1",
    )
    for feature in features:
        assert feature in dl1_features.columns

    # DL1B file as input
    assert (
        run_tool(
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
        == 1
    )


def test_stage1_datalevels(tmp_path):
    """test the dl1 tool on a file not providing r1, dl0 or dl1a"""
    from ctapipe.io import EventSource
    from ctapipe.tools.process import ProcessorTool

    class DummyEventSource(EventSource):
        @classmethod
        def is_compatible(cls, path):
            with open(path, "rb") as f:
                dummy = f.read(5)
                return dummy == b"dummy"

        @property
        def datalevels(self):
            return (DataLevel.R0,)

        @property
        def is_simulation(self):
            return True

        @property
        def obs_ids(self):
            return [1]

        @property
        def subarray(self):
            return None

        def _generator(self):
            return None

    dummy_file = tmp_path / "datalevels_dummy.h5"
    out_file = tmp_path / "datalevels_dummy_stage1_output.h5"
    with open(dummy_file, "wb") as f:
        f.write(b"dummy")
        f.flush()

    config = Path("./examples/stage1_config.json").absolute()
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
    """ check we can go to DL2 geometry from simtel file """
    config = Path("./examples/stage2_config.json").absolute()
    output = tmp_path / "test_stage2_from_simtel.DL2.h5"

    assert (
        run_tool(
            ProcessorTool(),
            argv=[
                f"--config={config}",
                f"--input={GAMMA_TEST_LARGE}",
                f"--output={output}",
                "--max-events=5",
                "--overwrite",
            ],
            cwd=tmp_path,
        )
        == 0
    )

    # check tables were written
    with tables.open_file(output, mode="r") as tf:
        assert tf.root.dl2.event.subarray.geometry.HillasReconstructor


def test_stage_2_from_dl1_images(tmp_path, dl1_image_file):
    """ check we can go to DL2 geometry from DL1 images """
    config = Path("./examples/stage2_config.json").absolute()
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
    with tables.open_file(output, mode="r") as tf:
        assert tf.root.dl2.event.subarray.geometry.HillasReconstructor


def test_stage_2_from_dl1_params(tmp_path, dl1_parameters_file):
    """ check we can go to DL2 geometry from DL1 parameters """

    config = Path("./examples/stage2_config.json").absolute()
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
    with tables.open_file(output, mode="r") as tf:
        assert tf.root.dl2.event.subarray.geometry.HillasReconstructor


def test_training_from_simtel(tmp_path):
    """ check we can write both dl1 and dl2 info (e.g. for training input) """

    config = Path("./examples/training_config.json").absolute()
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
            ],
            cwd=tmp_path,
        )
        == 0
    )

    # check tables were written
    with tables.open_file(output, mode="r") as tf:
        assert tf.root.dl1.event.telescope.parameters.tel_002
        assert tf.root.dl2.event.subarray.geometry.HillasReconstructor
