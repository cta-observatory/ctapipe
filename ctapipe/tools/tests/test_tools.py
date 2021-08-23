"""
Test individual tool functionality
"""
# pylint: disable=C0103,C0116,C0415
import subprocess
import sys

import matplotlib as mpl
from ctapipe.core import run_tool

from ctapipe.utils import get_dataset_path

GAMMA_TEST_LARGE = get_dataset_path("gamma_test_large.simtel.gz")


def test_display_summed_images(tmp_path):
    from ctapipe.tools.display_summed_images import ImageSumDisplayerTool

    mpl.use("Agg")
    assert (
        run_tool(
            ImageSumDisplayerTool(),
            argv=[f"--infile={GAMMA_TEST_LARGE}", "--max-events=2"],
            cwd=tmp_path,
        )
        == 0
    )

    assert run_tool(ImageSumDisplayerTool(), ["--help-all"]) == 0


def test_display_integrator(tmp_path):
    from ctapipe.tools.display_integrator import DisplayIntegrator

    mpl.use("Agg")

    assert (
        run_tool(
            DisplayIntegrator(),
            argv=[f"--input={GAMMA_TEST_LARGE}", "--max-events=1"],
            cwd=tmp_path,
        )
        == 0
    )

    assert run_tool(DisplayIntegrator(), ["--help-all"]) == 0


def test_display_events_single_tel(tmp_path):
    from ctapipe.tools.display_events_single_tel import SingleTelEventDisplay

    mpl.use("Agg")

    assert (
        run_tool(
            SingleTelEventDisplay(),
            argv=[
                f"--input={GAMMA_TEST_LARGE}",
                "--tel=11",
                "--max-events=2",  # <--- inconsistent!!!
            ],
            cwd=tmp_path,
        )
        == 0
    )

    assert run_tool(SingleTelEventDisplay(), ["--help-all"]) == 0


def test_display_dl1(tmp_path, dl1_image_file, dl1_parameters_file):
    from ctapipe.tools.display_dl1 import DisplayDL1Calib

    mpl.use("Agg")

    # test simtel
    assert (
        run_tool(
            DisplayDL1Calib(), argv=["--max-events=1", "--telescope=11"], cwd=tmp_path
        )
        == 0
    )
    # test DL1A
    assert (
        run_tool(
            DisplayDL1Calib(),
            argv=[f"--input={dl1_image_file}", "--max-events=1", "--telescope=11"],
        )
        == 0
    )
    # test DL1B, should error since nothing to plot
    ret = run_tool(
        DisplayDL1Calib(),
        argv=[f"--input={dl1_parameters_file}", "--max-events=1", "--telescope=11"],
    )
    assert ret == 1
    assert run_tool(DisplayDL1Calib(), ["--help-all"]) == 0


def test_info():
    from ctapipe.tools.info import info

    info(show_all=True)


def test_fileinfo(tmp_path, dl1_image_file):
    """ check we can run ctapipe-fileinfo and get results """
    import yaml
    from astropy.table import Table

    index_file = tmp_path / "index.fits"
    command = f"ctapipe-fileinfo {dl1_image_file} --output-table {index_file}"
    output = subprocess.run(command.split(" "), capture_output=True, check=True).stdout
    header = yaml.load(output)
    assert "ID" in header[str(dl1_image_file)]["CTA"]["ACTIVITY"]

    tab = Table.read(index_file)
    assert len(tab["CTA PRODUCT CREATION TIME"]) > 0

    command = f"ctapipe-fileinfo {dl1_image_file} --flat"
    output = subprocess.run(command.split(" "), capture_output=True, check=True).stdout
    header = yaml.load(output)
    assert "CTA ACTIVITY ID" in header[str(dl1_image_file)]


def test_dump_triggers(tmp_path):
    from ctapipe.tools.dump_triggers import DumpTriggersTool

    sys.argv = ["dump_triggers"]
    outfile = tmp_path / "triggers.fits"
    tool = DumpTriggersTool(infile=GAMMA_TEST_LARGE, outfile=str(outfile))

    assert run_tool(tool, cwd=tmp_path) == 0

    assert outfile.exists()
    assert run_tool(tool, ["--help-all"]) == 0


def test_dump_instrument(tmp_path):
    from ctapipe.tools.dump_instrument import DumpInstrumentTool

    sys.argv = ["dump_instrument"]
    tool = DumpInstrumentTool()

    assert run_tool(tool, [f"--input={GAMMA_TEST_LARGE}"], cwd=tmp_path) == 0
    assert (tmp_path / "FlashCam.camgeom.fits.gz").exists()

    assert (
        run_tool(tool, [f"--input={GAMMA_TEST_LARGE}", "--format=ecsv"], cwd=tmp_path)
        == 0
    )
    assert (tmp_path / "MonteCarloArray.optics.ecsv.txt").exists()

    assert (
        run_tool(tool, [f"--input={GAMMA_TEST_LARGE}", "--format=hdf5"], cwd=tmp_path)
        == 0
    )
    assert (tmp_path / "subarray.h5").exists()

    assert run_tool(tool, ["--help-all"], cwd=tmp_path) == 0


def test_camdemo(tmp_path):
    from ctapipe.tools.camdemo import CameraDemo

    sys.argv = ["camera_demo"]
    tool = CameraDemo()
    tool.num_events = 10
    tool.cleanframes = 2
    tool.display = False

    assert run_tool(tool, cwd=tmp_path) == 0
    assert run_tool(tool, ["--help-all"]) == 0


def test_bokeh_file_viewer(tmp_path):
    from ctapipe.tools.bokeh.file_viewer import BokehFileViewer

    sys.argv = ["bokeh_file_viewer"]
    tool = BokehFileViewer(disable_server=True)
    assert run_tool(tool, cwd=tmp_path) == 0
    assert tool.reader.input_url == get_dataset_path("gamma_test_large.simtel.gz")
    assert run_tool(tool, ["--help-all"]) == 0
