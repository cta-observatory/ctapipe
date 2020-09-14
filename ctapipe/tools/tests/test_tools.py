"""
Test individual tool functionality
"""

import os
import shlex
import sys

import matplotlib as mpl

import tempfile
import pandas as pd
import tables

from ctapipe.utils import get_dataset_path
from ctapipe.core import run_tool
from ctapipe.io import DataLevel
import numpy as np


GAMMA_TEST_LARGE = get_dataset_path("gamma_test_large.simtel.gz")
LST_MUONS = get_dataset_path("lst_muons.simtel.zst")


def test_stage_1():
    from ctapipe.tools.stage1 import Stage1ProcessorTool

    with tempfile.NamedTemporaryFile(suffix=".hdf5") as f:
        assert (
            run_tool(
                Stage1ProcessorTool(),
                argv=[
                    "--config=./examples/stage1_config.json",
                    f"--input={GAMMA_TEST_LARGE}",
                    f"--output={f.name}",
                    "--write-parameters",
                    "--overwrite",
                ],
            )
            == 0
        )

        # check tables were written
        with tables.open_file(f.name, mode="r") as tf:
            assert tf.root.dl1
            assert tf.root.dl1.event.telescope
            assert tf.root.dl1.event.subarray
            assert tf.root.configuration.instrument.subarray.layout
            assert tf.root.configuration.instrument.telescope.optics
            assert tf.root.configuration.instrument.telescope.camera.geometry_LSTCam
            assert tf.root.configuration.instrument.telescope.camera.readout_LSTCam

        # check we can read telescope parametrs
        dl1_features = pd.read_hdf(f.name, "/dl1/event/telescope/parameters/tel_001")
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

    with tempfile.NamedTemporaryFile(suffix=".hdf5") as f:
        assert (
            run_tool(
                Stage1ProcessorTool(),
                argv=[
                    "--config=./examples/stage1_config.json",
                    f"--input={GAMMA_TEST_LARGE}",
                    f"--output={f.name}",
                    "--write-images",
                    "--overwrite",
                ],
            )
            == 0
        )

        with tables.open_file(f.name, mode="r") as tf:
            assert tf.root.dl1
            assert tf.root.dl1.event.telescope
            assert tf.root.dl1.event.subarray
            assert tf.root.configuration.instrument.subarray.layout
            assert tf.root.configuration.instrument.telescope.optics
            assert tf.root.configuration.instrument.telescope.camera.geometry_LSTCam
            assert tf.root.configuration.instrument.telescope.camera.readout_LSTCam
            assert tf.root.dl1.event.telescope.images.tel_001
            dl1_image = tf.root.dl1.event.telescope.images.tel_001
            assert "image_mask" in dl1_image.dtype.names
            assert "image" in dl1_image.dtype.names
            assert "peak_time" in dl1_image.dtype.names


def test_stage1_datalevels():
    """test the dl1 tool on a file not providing r1 or dl0"""
    from ctapipe.io import EventSource
    from ctapipe.tools.stage1 import Stage1ProcessorTool

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
        def obs_id(self):
            return 1

        @property
        def subarray(self):
            return None

        def _generator(self):
            return None

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".dummy") as f:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".h5") as out:
            f.write(b"dummy")
            f.flush()

            tool = Stage1ProcessorTool()
            assert (
                run_tool(
                    tool,
                    argv=[
                        "--config=./examples/stage1_config.json",
                        f"--input={f.name}",
                        f"--output={out.name}",
                        "--write-images",
                        "--overwrite",
                    ],
                )
                == 1
            )
            # make sure the dummy event source was really used
            assert isinstance(tool.event_source, DummyEventSource)

            # we need to "touch" the output file again, otherwise tempfile will
            # complain it no longer exists as the tool removed it
            open(out.name, mode="a").close()


def test_muon_reconstruction(tmpdir):
    from ctapipe.tools.muon_reconstruction import MuonAnalysis

    with tempfile.NamedTemporaryFile(suffix=".hdf5") as f:
        assert (
            run_tool(
                MuonAnalysis(),
                argv=[f"--input={LST_MUONS}", f"--output={f.name}", "--overwrite"],
            )
            == 0
        )

        t = tables.open_file(f.name)
        table = t.root.dl1.event.telescope.parameters.muons[:]
        assert len(table) > 20
        assert np.count_nonzero(np.isnan(table["muonring_radius"])) == 0

    assert run_tool(MuonAnalysis(), ["--help-all"]) == 0


def test_display_summed_images(tmpdir):
    from ctapipe.tools.display_summed_images import ImageSumDisplayerTool

    mpl.use("Agg")
    assert (
        run_tool(
            ImageSumDisplayerTool(),
            argv=shlex.split(f"--infile={GAMMA_TEST_LARGE} " "--max-events=2 "),
        )
        == 0
    )

    assert run_tool(ImageSumDisplayerTool(), ["--help-all"]) == 0


def test_display_integrator(tmpdir):
    from ctapipe.tools.display_integrator import DisplayIntegrator

    mpl.use("Agg")

    assert (
        run_tool(
            DisplayIntegrator(),
            argv=shlex.split(f"--f={GAMMA_TEST_LARGE} " "--max_events=1 "),
        )
        == 0
    )

    assert run_tool(DisplayIntegrator(), ["--help-all"]) == 0


def test_display_events_single_tel(tmpdir):
    from ctapipe.tools.display_events_single_tel import SingleTelEventDisplay

    mpl.use("Agg")

    assert (
        run_tool(
            SingleTelEventDisplay(),
            argv=shlex.split(
                f"--infile={GAMMA_TEST_LARGE} "
                "--tel=11 "
                "--max-events=2 "  # <--- inconsistent!!!
            ),
        )
        == 0
    )

    assert run_tool(SingleTelEventDisplay(), ["--help-all"]) == 0


def test_display_dl1(tmpdir):
    from ctapipe.tools.display_dl1 import DisplayDL1Calib

    mpl.use("Agg")

    assert (
        run_tool(
            DisplayDL1Calib(), argv=shlex.split("--max_events=1 " "--telescope=11 ")
        )
        == 0
    )

    assert run_tool(DisplayDL1Calib(), ["--help-all"]) == 0


def test_info():
    from ctapipe.tools.info import info

    info(show_all=True)


def test_dump_triggers(tmpdir):
    from ctapipe.tools.dump_triggers import DumpTriggersTool

    sys.argv = ["dump_triggers"]
    outfile = tmpdir.join("triggers.fits")
    tool = DumpTriggersTool(infile=GAMMA_TEST_LARGE, outfile=str(outfile))

    assert run_tool(tool) == 0

    assert outfile.exists()
    assert run_tool(tool, ["--help-all"]) == 0


def test_dump_instrument(tmpdir):
    from ctapipe.tools.dump_instrument import DumpInstrumentTool

    sys.argv = ["dump_instrument"]
    tmpdir.chdir()

    tool = DumpInstrumentTool(infile=GAMMA_TEST_LARGE)

    assert run_tool(tool) == 0
    assert tmpdir.join("FlashCam.camgeom.fits.gz").exists()
    assert run_tool(tool, ["--help-all"]) == 0


def test_camdemo():
    from ctapipe.tools.camdemo import CameraDemo

    sys.argv = ["camera_demo"]
    tool = CameraDemo()
    tool.num_events = 10
    tool.cleanframes = 2
    tool.display = False

    assert run_tool(tool) == 0
    assert run_tool(tool, ["--help-all"]) == 0


def test_bokeh_file_viewer():
    from ctapipe.tools.bokeh.file_viewer import BokehFileViewer

    sys.argv = ["bokeh_file_viewer"]
    tool = BokehFileViewer(disable_server=True)
    assert run_tool(tool) == 0
    assert str(tool.reader.input_url) == get_dataset_path("gamma_test_large.simtel.gz")
    assert run_tool(tool, ["--help-all"]) == 0


def test_extract_charge_resolution(tmpdir):
    from ctapipe.tools.extract_charge_resolution import ChargeResolutionGenerator

    output_path = os.path.join(str(tmpdir), "cr.h5")
    tool = ChargeResolutionGenerator()
    assert run_tool(tool, ["-f", GAMMA_TEST_LARGE, "-O", output_path]) == 1
    # TODO: Test files do not contain true charge, cannot test tool fully
    # assert os.path.exists(output_path)
    assert run_tool(tool, ["--help-all"]) == 0


def test_plot_charge_resolution(tmpdir):
    from ctapipe.tools.plot_charge_resolution import ChargeResolutionViewer
    from ctapipe.plotting.tests.test_charge_resolution import create_temp_cr_file

    path = create_temp_cr_file(tmpdir)

    output_path = os.path.join(str(tmpdir), "cr.pdf")
    tool = ChargeResolutionViewer()

    assert run_tool(tool, ["-f", [path], "-o", output_path]) == 0
    assert os.path.exists(output_path)
    assert run_tool(tool, ["--help-all"]) == 0
