import sys
from ctapipe.tools.camdemo import CameraDemo
from ctapipe.tools.dump_triggers import DumpTriggersTool
from ctapipe.tools.dump_instrument import DumpInstrumentTool
from ctapipe.tools.info import info
from ctapipe.tools.bokeh.file_viewer import BokehFileViewer
from ctapipe.tools.extract_charge_resolution import ChargeResolutionGenerator
from ctapipe.tools.plot_charge_resolution import ChargeResolutionViewer
from ctapipe.utils import get_dataset_path
import os
import pytest


def test_info():
    info(show_all=True)


def test_dump_triggers(tmpdir):
    sys.argv = ['dump_triggers']
    outfile = tmpdir.join("triggers.fits")

    tool = DumpTriggersTool(
        infile=get_dataset_path("gamma_test_large.simtel.gz"),
        outfile=str(outfile)
    )

    tool.run(argv=[])

    assert outfile.exists()


def test_dump_instrument(tmpdir):
    sys.argv = ['dump_instrument']
    tmpdir.chdir()

    tool = DumpInstrumentTool(
        infile=get_dataset_path("gamma_test_large.simtel.gz"),
    )

    tool.run(argv=[])

    print(tmpdir.listdir())
    assert tmpdir.join('FlashCam.camgeom.fits.gz').exists()


def test_camdemo():
    sys.argv = ['camera_demo']
    tool = CameraDemo()
    tool.num_events = 10
    tool.cleanframes = 2
    tool.display = False
    tool.run(argv=[])


def test_bokeh_file_viewer():
    sys.argv = ['bokeh_file_viewer']
    tool = BokehFileViewer(disable_server=True)
    tool.run()

    assert tool.reader.input_url == get_dataset_path("gamma_test.simtel.gz")


def test_extract_charge_resolution(tmpdir):
    output_path = os.path.join(str(tmpdir), "cr.h5")
    tool = ChargeResolutionGenerator()
    with pytest.raises(KeyError):
        tool.run([
            '-f', get_dataset_path("gamma_test_large.simtel.gz"),
            '-o', output_path,
        ])
    # TODO: Test files do not contain true charge, cannot test tool fully
    # assert os.path.exists(output_path)


def test_plot_charge_resolution(tmpdir):
    from ctapipe.plotting.tests.test_charge_resolution import \
        create_temp_cr_file
    path = create_temp_cr_file(tmpdir)

    output_path = os.path.join(str(tmpdir), "cr.pdf")
    tool = ChargeResolutionViewer()
    tool.run([
        '-f', [path],
        '-o', output_path,
    ])
    assert os.path.exists(output_path)
