import os
import shlex
import sys

import matplotlib as mpl
import pytest

from ctapipe.utils import get_dataset_path

GAMMA_TEST_LARGE = get_dataset_path("gamma_test_large.simtel.gz")


def test_muon_reconstruction(tmpdir):
    from ctapipe.tools.muon_reconstruction import MuonDisplayerTool
    tool = MuonDisplayerTool()
    tool.run(
        argv=shlex.split(
            f'--input={GAMMA_TEST_LARGE} '
            '--max_events=2 '
        )
    )

    with pytest.raises(SystemExit):
        tool.run(['--help-all'])


def test_display_summed_images(tmpdir):
    from ctapipe.tools.display_summed_images import ImageSumDisplayerTool
    mpl.use('Agg')
    tool = ImageSumDisplayerTool()
    tool.run(
        argv=shlex.split(
            f'--infile={GAMMA_TEST_LARGE} '
            '--max-events=2 '
        )
    )

    with pytest.raises(SystemExit):
        tool.run(['--help-all'])


def test_display_integrator(tmpdir):
    from ctapipe.tools.display_integrator import DisplayIntegrator
    mpl.use('Agg')
    tool = DisplayIntegrator()
    tool.run(
        argv=shlex.split(
            f'--f={GAMMA_TEST_LARGE} '
            '--max_events=1 '
        )
    )

    with pytest.raises(SystemExit):
        tool.run(['--help-all'])


def test_display_events_single_tel(tmpdir):
    from ctapipe.tools.display_events_single_tel import SingleTelEventDisplay
    mpl.use('Agg')
    tool = SingleTelEventDisplay()
    tool.run(
        argv=shlex.split(
            f'--infile={GAMMA_TEST_LARGE} '
            '--tel=11 '
            '--max-events=2 '  # <--- inconsistent!!!
        )
    )

    with pytest.raises(SystemExit):
        tool.run(['--help-all'])


def test_display_dl1(tmpdir):
    from ctapipe.tools.display_dl1 import DisplayDL1Calib
    mpl.use('Agg')
    tool = DisplayDL1Calib()
    tool.run(
        argv=shlex.split(
            '--max_events=1 '
            '--telescope=11 '
        )
    )

    with pytest.raises(SystemExit):
        tool.run(['--help-all'])


def test_info():
    from ctapipe.tools.info import info
    info(show_all=True)


def test_dump_triggers(tmpdir):
    from ctapipe.tools.dump_triggers import DumpTriggersTool

    sys.argv = ['dump_triggers']
    outfile = tmpdir.join("triggers.fits")

    tool = DumpTriggersTool(
        infile=GAMMA_TEST_LARGE,
        outfile=str(outfile)
    )

    tool.run(argv=[])

    assert outfile.exists()

    with pytest.raises(SystemExit):
        tool.run(['--help-all'])


def test_dump_instrument(tmpdir):
    from ctapipe.tools.dump_instrument import DumpInstrumentTool

    sys.argv = ['dump_instrument']
    tmpdir.chdir()

    tool = DumpInstrumentTool(
        infile=GAMMA_TEST_LARGE,
    )

    tool.run(argv=[])

    print(tmpdir.listdir())
    assert tmpdir.join('FlashCam.camgeom.fits.gz').exists()

    with pytest.raises(SystemExit):
        tool.run(['--help-all'])


def test_camdemo():
    from ctapipe.tools.camdemo import CameraDemo
    sys.argv = ['camera_demo']
    tool = CameraDemo()
    tool.num_events = 10
    tool.cleanframes = 2
    tool.display = False
    tool.run(argv=[])

    with pytest.raises(SystemExit):
        tool.run(['--help-all'])


def test_bokeh_file_viewer():
    from ctapipe.tools.bokeh.file_viewer import BokehFileViewer

    sys.argv = ['bokeh_file_viewer']
    tool = BokehFileViewer(disable_server=True)
    tool.run()

    assert tool.reader.input_url == get_dataset_path("gamma_test_large.simtel.gz")

    with pytest.raises(SystemExit):
        tool.run(['--help-all'])


def test_extract_charge_resolution(tmpdir):
    from ctapipe.tools.extract_charge_resolution import (
        ChargeResolutionGenerator
    )

    output_path = os.path.join(str(tmpdir), "cr.h5")
    tool = ChargeResolutionGenerator()
    with pytest.raises(KeyError):
        tool.run([
            '-f', GAMMA_TEST_LARGE,
            '-O', output_path,
        ])
    # TODO: Test files do not contain true charge, cannot test tool fully
    # assert os.path.exists(output_path)

    with pytest.raises(SystemExit):
        tool.run(['--help-all'])


def test_plot_charge_resolution(tmpdir):
    from ctapipe.tools.plot_charge_resolution import ChargeResolutionViewer
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

    with pytest.raises(SystemExit):
        tool.run(['--help-all'])
