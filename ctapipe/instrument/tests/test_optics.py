""" Tests for OpticsDescriptions"""
import pytest
from astropy import units as u
import os
import tempfile

from ctapipe.instrument.optics import OpticsDescription

from ctapipe.utils import get_table_dataset
from ctapipe.core.tool import run_tool
from ctapipe.tools.dump_instrument import DumpInstrumentTool
from ctapipe.utils.datasets import get_dataset_path


def test_guess_optics():
    """ make sure we can guess an optics type from metadata"""
    from ctapipe.instrument import guess_telescope

    answer = guess_telescope(1855, 28.0 * u.m)

    od = OpticsDescription.from_name(answer.name)

    assert od.equivalent_focal_length.to_value(u.m) == 28
    assert od.num_mirrors == 1


def test_construct_optics():
    """create an OpticsDescription and make sure it
    fails if units are missing"""
    OpticsDescription(
        name="test",
        num_mirrors=1,
        num_mirror_tiles=100,
        mirror_area=u.Quantity(550, u.m ** 2),
        equivalent_focal_length=u.Quantity(10, u.m),
    )

    with pytest.raises(TypeError):
        OpticsDescription(
            name="test",
            num_mirrors=1,
            num_mirror_tiles=100,
            mirror_area=550,
            equivalent_focal_length=10,
        )


@pytest.mark.parametrize("optics_name", OpticsDescription.get_known_optics_names())
def test_optics_from_name(optics_name):
    """ try constructing all by name """
    optics = OpticsDescription.from_name(optics_name)
    assert optics.equivalent_focal_length > 0
    # make sure the string rep gives back the name:
    assert str(optics) == optics_name


def test_optics_from_name_user_supplied_table():
    table = get_table_dataset("optics", role="")
    optics = OpticsDescription.from_name("SST-GCT", optics_table=table)
    assert optics.name == "SST-GCT"
    assert optics.mirror_area > 1.0 * u.m ** 2


def test_optics_from_dump_instrument():
    # test with file written by dump-instrument

    svc_path_before = os.getenv("CTAPIPE_SVC_PATH")
    cwd = os.getcwd()

    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        os.environ["CTAPIPE_SVC_PATH"] = tmp_dir

        infile = get_dataset_path("gamma_test_large.simtel.gz")
        run_tool(DumpInstrumentTool(), [f"--input={infile}", "--format=ecsv"])

        lst = OpticsDescription.from_name("LST_LST_LSTCam", "MonteCarloArray.optics")
        assert lst.num_mirrors == 1
        assert lst.equivalent_focal_length.to_value(u.m) == 28
        assert lst.num_mirror_tiles == 198

    os.chdir(cwd)
    if svc_path_before is None:
        del os.environ["CTAPIPE_SVC_PATH"]
    else:
        os.environ["CTAPIPE_SVC_PATH"] = svc_path_before
