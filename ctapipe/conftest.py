"""
common pytest fixtures for tests in ctapipe
"""

from copy import deepcopy

import pytest

from ctapipe.instrument import CameraGeometry
from ctapipe.io import SimTelEventSource
from ctapipe.utils import get_dataset_path
from ctapipe.utils.filelock import FileLock

# names of camera geometries available on the data server
camera_names = [
    "ASTRICam",
    "CHEC",
    "DigiCam",
    "FACT",
    "FlashCam",
    "HESS-I",
    "HESS-II",
    "LSTCam",
    "MAGICCam",
    "NectarCam",
    "SCTCam",
    "VERITAS",
    "Whipple490",
]


@pytest.fixture(scope="function", params=camera_names)
def camera_geometry(request):
    return CameraGeometry.from_name(request.param)


@pytest.fixture(scope="session")
def _global_example_event():
    """
    helper to get a single event from a MC file. Don't use this fixture
    directly, rather use `test_event`
    """
    filename = get_dataset_path("gamma_test_large.simtel.gz")

    print("******************** LOAD TEST EVENT ***********************")

    with SimTelEventSource(input_url=filename) as reader:
        event = next(iter(reader))

    return event


@pytest.fixture(scope="session")
def example_subarray():
    """
    Subarray corresponding to the example event
    """
    filename = get_dataset_path("gamma_test_large.simtel.gz")

    print("******************** LOAD TEST EVENT ***********************")

    with SimTelEventSource(input_url=filename) as reader:
        return reader.subarray


@pytest.fixture(scope="function")
def example_event(_global_example_event):
    """
    Use this fixture anywhere you need a test event read from a MC file. For
    example:

    .. code-block::
        def test_my_thing(test_event):
            assert len(test_event.r0.tel) > 0

    """
    return deepcopy(_global_example_event)


@pytest.fixture(scope="session")
def _subarray_and_event_gamma_off_axis_500_gev():
    from ctapipe.calib import CameraCalibrator
    from ctapipe.image import ImageProcessor

    path = get_dataset_path("lst_prod3_calibration_and_mcphotons.simtel.zst")

    with SimTelEventSource(path) as source:
        it = iter(source)
        # we want the second event, first event is a corner clipper
        next(it)
        event = next(it)

        # make dl1a available
        calib = CameraCalibrator(source.subarray)
        calib(event)

        image_processor = ImageProcessor(source.subarray)

        # make dl1b available
        image_processor(event)
        return source.subarray, event


@pytest.fixture(scope="function")
def subarray_and_event_gamma_off_axis_500_gev(
    _subarray_and_event_gamma_off_axis_500_gev,
):
    """
    A four LST subarray event with a nice shower, well suited to test
    reconstruction algorithms.

    This event should be very well reconstructible, as we have four LSTs with
    bright events.

    The event is already calibrated and image parameters have been calculated.

    You can safely mutate the event or subarray in a test as each test
    gets a fresh copy.
    """
    subarray, event = _subarray_and_event_gamma_off_axis_500_gev
    return deepcopy(subarray), deepcopy(event)


@pytest.fixture(scope="session")
def prod5_gamma_simtel_path():
    return get_dataset_path(
        "gamma_20deg_0deg_run2___cta-prod5-paranal_desert-2147m-Paranal-dark_cone10-100evts.simtel.zst"
    )


@pytest.fixture(scope="session")
def prod5_proton_simtel_path():
    return get_dataset_path(
        "proton_20deg_0deg_run4___cta-prod5-paranal_desert-2147m-Paranal-dark-100evts.simtel.zst"
    )


@pytest.fixture(scope="session")
def dl1_tmp_path(tmp_path_factory):
    """Temporary directory for global dl1 test data"""
    return tmp_path_factory.mktemp("dl1")


@pytest.fixture(scope="session")
def dl2_tmp_path(tmp_path_factory):
    """Temporary directory for global dl2 test data"""
    return tmp_path_factory.mktemp("dl2")


@pytest.fixture(scope="session")
def dl2_shower_geometry_file(dl2_tmp_path, prod5_gamma_simtel_path):
    """
    File containing both parameters and shower geometry from a gamma simulation set.
    """
    from ctapipe.core import run_tool
    from ctapipe.tools.process import ProcessorTool

    output = dl2_tmp_path / "gamma.training.h5"

    # prevent running process multiple times in case of parallel tests
    with FileLock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file():
            return output

        argv = [
            f"--input={prod5_gamma_simtel_path}",
            f"--output={output}",
            "--write-images",
            "--write-stereo-shower",
            "--max-events=20",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl2_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl2_proton_geometry_file(dl2_tmp_path, prod5_proton_simtel_path):
    """
    File containing both parameters and shower geometry from a gamma simulation set.
    """
    from ctapipe.core import run_tool
    from ctapipe.tools.process import ProcessorTool

    output = dl2_tmp_path / "proton.training.h5"

    # prevent running process multiple times in case of parallel tests
    with FileLock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file():
            return output

        argv = [
            f"--input={prod5_proton_simtel_path}",
            f"--output={output}",
            "--write-images",
            "--write-stereo-shower",
            "--max-events=20",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl2_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl2_shower_geometry_file_type(dl2_tmp_path, prod5_gamma_simtel_path):
    """
    File containing both parameters and shower geometry from a gamma simulation set.
    """
    from ctapipe.core import run_tool
    from ctapipe.tools.process import ProcessorTool

    output = dl2_tmp_path / "gamma_by_type.training.h5"

    # prevent running process multiple times in case of parallel tests
    with FileLock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file():
            return output

        argv = [
            f"--input={prod5_gamma_simtel_path}",
            f"--output={output}",
            "--write-images",
            "--write-stereo-shower",
            "--max-events=20",
            "--DataWriter.split_datasets_by=tel_type",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl2_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl1_file(dl1_tmp_path, prod5_gamma_simtel_path):
    """
    DL1 file containing both images and parameters from a gamma simulation set.
    """
    from ctapipe.core import run_tool
    from ctapipe.tools.process import ProcessorTool

    output = dl1_tmp_path / "gamma.dl1.h5"

    # prevent running process multiple times in case of parallel tests
    with FileLock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file():
            return output

        argv = [
            f"--input={prod5_gamma_simtel_path}",
            f"--output={output}",
            "--write-images",
            "--max-events=20",
            "--DataWriter.Contact.name=αℓℓ the äüöß",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl1_by_type_file(dl1_tmp_path, prod5_gamma_simtel_path):
    """
    DL1 file containing both images and parameters from a gamma simulation set.
    """
    from ctapipe.tools.process import ProcessorTool
    from ctapipe.core import run_tool

    output = dl1_tmp_path / "gamma_by_type.dl1.h5"

    # prevent running stage1 multiple times in case of parallel tests
    with FileLock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file():
            return output

        argv = [
            f"--input={prod5_gamma_simtel_path}",
            f"--output={output}",
            "--write-images",
            "--max-events=20",
            "--DataWriter.split_datasets_by=tel_type",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl1_image_file(dl1_tmp_path, prod5_gamma_simtel_path):
    """
    DL1 file containing only images (DL1A) from a gamma simulation set.
    """
    from ctapipe.core import run_tool
    from ctapipe.tools.process import ProcessorTool

    output = dl1_tmp_path / "gamma_images.dl1.h5"

    # prevent running process multiple times in case of parallel tests
    with FileLock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file():
            return output

        argv = [
            f"--input={prod5_gamma_simtel_path}",
            f"--output={output}",
            "--write-images",
            "--DataWriter.write_parameters=False",
            "--max-events=20",
            "--DataWriter.Contact.name=αℓℓ the äüöß",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl1_parameters_file(dl1_tmp_path, prod5_gamma_simtel_path):
    """
    DL1 File containing only parameters (DL1B) from a gamma simulation set.
    """
    from ctapipe.core import run_tool
    from ctapipe.tools.process import ProcessorTool

    output = dl1_tmp_path / "gamma_parameters.dl1.h5"

    # prevent running process multiple times in case of parallel tests
    with FileLock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file():
            return output

        argv = [
            f"--input={prod5_gamma_simtel_path}",
            f"--output={output}",
            "--write-parameters",
            "--max-events=20",
            "--DataWriter.Contact.name=αℓℓ the äüöß",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl1_muon_file(dl1_tmp_path):
    """
    DL1 file containing only images from a muon simulation set.
    """
    from ctapipe.core import run_tool
    from ctapipe.tools.process import ProcessorTool

    output = dl1_tmp_path / "muons.dl1.h5"

    # prevent running process multiple times in case of parallel tests
    with FileLock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file():
            return output

        infile = get_dataset_path("lst_muons.simtel.zst")
        argv = [
            f"--input={infile}",
            f"--output={output}",
            "--write-images",
            "--DataWriter.write_parameters=False",
            "--DataWriter.Contact.name=αℓℓ the äüöß",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl1_proton_file(dl1_tmp_path, prod5_proton_simtel_path):
    """
    DL1 file containing images and parameters for a prod5 proton run
    """
    from ctapipe.tools.process import ProcessorTool
    from ctapipe.core import run_tool

    output = dl1_tmp_path / "proton.dl1.h5"

    with FileLock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file():
            return output

        argv = [
            f"--input={prod5_proton_simtel_path}",
            f"--output={output}",
            "--write-images",
            "--DataWriter.Contact.name=αℓℓ the äüöß",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
        return output
