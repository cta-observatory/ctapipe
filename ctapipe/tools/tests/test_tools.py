"""
Test individual tool functionality
"""

import os
import shlex
import sys
import subprocess
import pytest

import matplotlib as mpl

import tempfile
import pandas as pd
import tables

from ctapipe.utils import get_dataset_path
from ctapipe.core import run_tool
from ctapipe.io import DataLevel
import numpy as np
from pathlib import Path


tmp_dir = tempfile.TemporaryDirectory()
GAMMA_TEST_LARGE = get_dataset_path("gamma_test_large.simtel.gz")
LST_MUONS = get_dataset_path("lst_muons.simtel.zst")


@pytest.fixture(scope="module")
def dl1_image_file():
    """
    DL1 file containing only images (DL1A) from a gamma simulation set.
    """
    command = (
        "ctapipe-stage1 "
        f"--input {GAMMA_TEST_LARGE} "
        f"--output {tmp_dir.name}/images.dl1.h5 "
        "--write-images "
        "--max-events 20 "
        "--allowed-tels=[1,2,3]"
    )
    subprocess.call(command.split(), stdout=subprocess.PIPE)
    return f"{tmp_dir.name}/images.dl1.h5"


@pytest.fixture(scope="module")
def dl1_parameters_file():
    """
    DL1 File containing only parameters (DL1B) from a gamma simulation set.
    """
    command = (
        "ctapipe-stage1 "
        f"--input {GAMMA_TEST_LARGE} "
        f"--output {tmp_dir.name}/parameters.dl1.h5 "
        "--write-parameters "
        "--max-events 20 "
        "--allowed-tels=[1,2,3]"
    )
    subprocess.call(command.split(), stdout=subprocess.PIPE)
    return f"{tmp_dir.name}/parameters.dl1.h5"


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


def test_merge(tmpdir):
    from ctapipe.tools.dl1_merge import MergeTool
    from ctapipe.tools.stage1 import Stage1Tool

    config = Path("./examples/stage1_config.json").absolute()

    tmp_dir = tempfile.TemporaryDirectory()
    in_1 = tmp_dir.name + "/test_file_1.hdf5"
    in_2 = tmp_dir.name + "/test_file_2.hdf5"
    out_all = tmp_dir.name + "/merged_file_all.hdf5"
    out_skip_images = tmp_dir.name + "/merged_file_images.hdf5"
    out_skip_parameters = tmp_dir.name + "/merged_file_parameters.hdf5"
    out_tels_dir_pattern = tmp_dir.name + "/merged_file_tels_dir_pattern.hdf5"

    assert (
        run_tool(
            Stage1Tool(),
            argv=[
                f"--config={config}",
                f"--input={GAMMA_TEST_LARGE}",
                f"--output={in_1}",
                "--write-parameters",
                "--write-images",
                "--overwrite",
            ],
            cwd=tmpdir,
        )
        == 0
    )
    assert (
        run_tool(
            Stage1Tool(),
            argv=[
                f"--config={config}",
                f"--input={GAMMA_TEST_LARGE}",
                f"--output={in_2}",
                "--write-parameters",
                "--write-images",
                "--overwrite",
            ],
            cwd=tmpdir,
        )
        == 0
    )

    assert (
        run_tool(
            MergeTool(),
            argv=[
                f"--i={tmp_dir.name}",
                "--p=test_file_*.hdf5",
                f"--o={out_tels_dir_pattern}",
                "--overwrite",
                "--t=[2, 3]",
            ],
            cwd=tmpdir,
        )
        == 0
    )

    assert (
        run_tool(
            MergeTool(), argv=[in_1, in_2, f"--o={out_all}", "--overwrite"], cwd=tmpdir
        )
        == 0
    )

    assert (
        run_tool(
            MergeTool(),
            argv=[in_1, in_2, f"--o={out_skip_images}", "--overwrite", "--skip-images"],
            cwd=tmpdir,
        )
        == 0
    )

    assert (
        run_tool(
            MergeTool(),
            argv=[
                in_1,
                in_2,
                f"--o={out_skip_parameters}",
                "--overwrite",
                "--skip-parameters",
            ],
            cwd=tmpdir,
        )
        == 0
    )

    out_files_list = [
        out_all,
        out_skip_images,
        out_skip_parameters,
        out_tels_dir_pattern,
    ]

    for out_file in out_files_list:
        with tables.open_file(out_file, mode="r") as out_f, tables.open_file(
            in_1, mode="r"
        ) as in_f:

            # Check expanded tables
            assert len(out_f.root.simulation.service.shower_distribution) == 2
            assert len(out_f.root.simulation.event.subarray.shower) == 220
            assert len(out_f.root.configuration.simulation.run) == 2
            assert len(out_f.root.dl1.monitoring.subarray.pointing) == 2
            assert len(out_f.root.dl1.event.subarray.trigger) == 220
            assert len(out_f.root.dl1.event.telescope.trigger) == 918
            assert len(out_f.root.simulation.service.shower_distribution) == 2
            # Check subarray and service meta
            assert out_f.root.dl1.service["image_statistics.__table_column_meta__"]
            assert out_f.root.configuration.instrument.subarray.layout
            assert out_f.root.configuration.instrument.telescope.optics
            assert out_f.root.configuration.instrument.telescope.camera.geometry_LSTCam
            assert out_f.root.configuration.instrument.telescope.camera.readout_LSTCam

            # Check image statistics
            table_in = in_f.root["/dl1/service/image_statistics"]
            table_out = out_f.root["/dl1/service/image_statistics"]
            for row in range(len(table_in)):
                assert table_out.cols.counts[row] == np.multiply(
                    table_in.cols.counts[row], 2
                )
                assert table_out.cols.cumulative_counts[row] == np.multiply(
                    table_in.cols.cumulative_counts[row], 2
                )

            # Check telescope tables
            if out_file == out_tels_dir_pattern:
                telescope_nodes = {
                    "/dl1/monitoring/telescope/pointing",
                    "/dl1/event/telescope/images",
                    "/dl1/event/telescope/parameters",
                }
                for node in telescope_nodes:
                    assert len(out_f.list_nodes(node)) == 2
                    for tel_name in {"tel_002", "tel_003"}:
                        assert len(out_f.root[node + "/" + tel_name]) == np.multiply(
                            len(in_f.root[node + "/" + tel_name]), 2
                        )
                continue

            for tel in in_f.root.dl1.monitoring.telescope.pointing:
                assert len(
                    out_f.root.dl1.monitoring.telescope.pointing[tel.name]
                ) == np.multiply(
                    len(in_f.root.dl1.monitoring.telescope.pointing[tel.name]), 2
                )

            if out_file != out_skip_images:
                for tel in in_f.root.dl1.event.telescope.images:
                    assert len(
                        out_f.root.dl1.event.telescope.images[tel.name]
                    ) == np.multiply(
                        len(in_f.root.dl1.event.telescope.images[tel.name]), 2
                    )

            if out_file != out_skip_parameters:
                for tel in in_f.root.dl1.event.telescope.parameters:
                    assert len(
                        out_f.root.dl1.event.telescope.parameters[tel.name]
                    ) == np.multiply(
                        len(in_f.root.dl1.event.telescope.parameters[tel.name]), 2
                    )

    config = Path("./examples/stage1_config.json").absolute()
    dl1b_file = tmp_dir.name + "/dl1b_from_simtel.dl1.h5"
    assert (
        run_tool(
            Stage1Tool(),
            argv=[
                f"--config={config}",
                f"--input={GAMMA_TEST_LARGE}",
                f"--output={dl1b_file}",
                "--write-parameters",
                "--overwrite",
            ],
            cwd=tmpdir,
        )
        == 0
    )

    # check tables were written
    with tables.open_file(dl1b_file, mode="r") as tf:
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
    dl1_features = pd.read_hdf(dl1b_file, "/dl1/event/telescope/parameters/tel_001")
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

    dl1a_file = tmp_dir.name + "/dl1a_from_simtel.dl1.h5"
    assert (
        run_tool(
            Stage1Tool(),
            argv=[
                f"--config={config}",
                f"--input={GAMMA_TEST_LARGE}",
                f"--output={dl1a_file}",
                "--write-images",
                "--overwrite",
            ],
            cwd=tmpdir,
        )
        == 0
    )

    with tables.open_file(dl1a_file, mode="r") as tf:
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


def test_stage_1_dl1(tmpdir, dl1_image_file, dl1_parameters_file):
    from ctapipe.tools.stage1 import Stage1Tool

    config = Path("./examples/stage1_config.json").absolute()
    # DL1A file as input
    dl1b_from_dl1a_file = tmp_dir.name + "/dl1b_from dl1a.dl1.h5"
    assert (
        run_tool(
            Stage1Tool(),
            argv=[
                f"--config={config}",
                f"--input={dl1_image_file}",
                f"--output={dl1b_from_dl1a_file}",
                "--write-parameters",
                "--overwrite",
            ],
            cwd=tmpdir,
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
        dl1b_from_dl1a_file, "/dl1/event/telescope/parameters/tel_001"
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
            Stage1Tool(),
            argv=[
                f"--config={config}",
                f"--input={dl1_parameters_file}",
                f"--output={tmp_dir.name + '/dl1b_from_dl1b.dl1.h5'}",
                "--write-parameters",
                "--overwrite",
            ],
            cwd=tmpdir,
        )
        == 1
    )


def test_stage1_datalevels(tmpdir):
    """test the dl1 tool on a file not providing r1, dl0 or dl1a"""
    from ctapipe.io import EventSource
    from ctapipe.tools.stage1 import Stage1Tool

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

    dummy_file = tmp_dir.name + "/datalevels_dummy.h5"
    out_file = tmp_dir.name + "/datalevels_dummy_stage1_output.h5"
    with open(dummy_file, "wb") as f:
        f.write(b"dummy")
        f.flush()

    config = Path("./examples/stage1_config.json").absolute()
    tool = Stage1Tool()

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
            cwd=tmpdir,
        )
        == 1
    )
    # make sure the dummy event source was really used
    assert isinstance(tool.event_source, DummyEventSource)


def test_muon_reconstruction(tmpdir, dl1_muon_file):
    from ctapipe.tools.muon_reconstruction import MuonAnalysis

    muon_simtel_output_file = tmp_dir.name + "/muon_reco_on_simtel.h5"
    assert (
        run_tool(
            MuonAnalysis(),
            argv=[
                f"--input={LST_MUONS}",
                f"--output={muon_simtel_output_file}",
                "--overwrite",
            ],
            cwd=tmpdir,
        )
        == 0
    )

    with tables.open_file(muon_simtel_output_file) as t:
        table = t.root.dl1.event.telescope.parameters.muons[:]
        assert len(table) > 20
        assert np.count_nonzero(np.isnan(table["muonring_radius"])) == 0

    muon_dl1_output_file = tmp_dir.name + "/muon_reco_on_dl1a.h5"
    assert (
        run_tool(
            MuonAnalysis(),
            argv=[
                f"--input={dl1_muon_file}",
                f"--output={muon_dl1_output_file}",
                "--overwrite",
            ],
            cwd=tmpdir,
        )
        == 0
    )

    with tables.open_file(muon_dl1_output_file) as t:
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
            cwd=tmpdir,
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
            cwd=tmpdir,
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
                f"--input={GAMMA_TEST_LARGE} "
                "--tel=11 "
                "--max-events=2 "  # <--- inconsistent!!!
            ),
            cwd=tmpdir,
        )
        == 0
    )

    assert run_tool(SingleTelEventDisplay(), ["--help-all"]) == 0


def test_display_dl1(tmpdir, dl1_image_file, dl1_parameters_file):
    from ctapipe.tools.display_dl1 import DisplayDL1Calib

    mpl.use("Agg")

    # test simtel
    assert (
        run_tool(
            DisplayDL1Calib(),
            argv=shlex.split("--max_events=1 " "--telescope=11 "),
            cwd=tmpdir,
        )
        == 0
    )
    # test DL1A
    assert (
        run_tool(
            DisplayDL1Calib(),
            argv=shlex.split(
                f"--input {dl1_image_file} --max_events=1 " "--telescope=11 "
            ),
        )
        == 0
    )
    # test DL1B
    assert (
        run_tool(
            DisplayDL1Calib(),
            argv=shlex.split(
                f"--input {dl1_parameters_file} --max_events=1 " "--telescope=11 "
            ),
        )
        == 1
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

    assert run_tool(tool, cwd=tmpdir) == 0

    assert outfile.exists()
    assert run_tool(tool, ["--help-all"]) == 0


def test_dump_instrument(tmpdir):
    from ctapipe.tools.dump_instrument import DumpInstrumentTool

    sys.argv = ["dump_instrument"]
    tmpdir.chdir()

    tool = DumpInstrumentTool()

    assert run_tool(tool, [f"--input={GAMMA_TEST_LARGE}"], cwd=tmpdir) == 0
    assert tmpdir.join("FlashCam.camgeom.fits.gz").exists()

    assert (
        run_tool(tool, [f"--input={GAMMA_TEST_LARGE}", "--format=ecsv"], cwd=tmpdir)
        == 0
    )
    assert tmpdir.join("MonteCarloArray.optics.ecsv.txt").exists()

    assert (
        run_tool(tool, [f"--input={GAMMA_TEST_LARGE}", "--format=hdf5"], cwd=tmpdir)
        == 0
    )
    assert tmpdir.join("subarray.h5").exists()

    assert run_tool(tool, ["--help-all"], cwd=tmpdir) == 0


def test_camdemo(tmpdir):
    from ctapipe.tools.camdemo import CameraDemo

    sys.argv = ["camera_demo"]
    tool = CameraDemo()
    tool.num_events = 10
    tool.cleanframes = 2
    tool.display = False

    assert run_tool(tool, cwd=tmpdir) == 0
    assert run_tool(tool, ["--help-all"]) == 0


def test_bokeh_file_viewer(tmpdir):
    from ctapipe.tools.bokeh.file_viewer import BokehFileViewer

    sys.argv = ["bokeh_file_viewer"]
    tool = BokehFileViewer(disable_server=True)
    assert run_tool(tool, cwd=tmpdir) == 0
    assert tool.reader.input_url == get_dataset_path("gamma_test_large.simtel.gz")
    assert run_tool(tool, ["--help-all"]) == 0


def test_extract_charge_resolution(tmpdir):
    from ctapipe.tools.extract_charge_resolution import ChargeResolutionGenerator

    output_path = os.path.join(str(tmpdir), "cr.h5")
    tool = ChargeResolutionGenerator()

    assert (
        run_tool(tool, ["-f", str(GAMMA_TEST_LARGE), "-O", output_path], cwd=tmpdir)
        == 1
    )
    # TODO: Test files do not contain true charge, cannot test tool fully
    # assert os.path.exists(output_path)
    assert run_tool(tool, ["--help-all"]) == 0


def test_plot_charge_resolution(tmpdir):
    from ctapipe.tools.plot_charge_resolution import ChargeResolutionViewer
    from ctapipe.plotting.tests.test_charge_resolution import create_temp_cr_file

    path = create_temp_cr_file(tmpdir)

    output_path = os.path.join(str(tmpdir), "cr.pdf")
    tool = ChargeResolutionViewer()

    argv = ["-f", str(path), "-o", output_path]
    assert run_tool(tool, argv) == 0
    assert os.path.exists(output_path)
    assert run_tool(tool, ["--help-all"]) == 0
