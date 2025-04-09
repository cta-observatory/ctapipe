"""
common pytest fixtures for tests in ctapipe
"""
import shutil
from copy import deepcopy

import astropy.units as u
import numpy as np
import pytest
import tables
from astropy.coordinates import EarthLocation
from astropy.table import Table
from pytest_astropy_header.display import PYTEST_HEADER_MODULES

from ctapipe.core import run_tool
from ctapipe.instrument import CameraGeometry, FromNameWarning, SubarrayDescription
from ctapipe.io import SimTelEventSource
from ctapipe.utils import get_dataset_path
from ctapipe.utils.datasets import resource_file
from ctapipe.utils.filelock import FileLock

PYTEST_HEADER_MODULES.clear()
PYTEST_HEADER_MODULES["eventio"] = "eventio"
PYTEST_HEADER_MODULES["numpy"] = "numpy"
PYTEST_HEADER_MODULES["scipy"] = "scipy"
PYTEST_HEADER_MODULES["astropy"] = "astropy"
PYTEST_HEADER_MODULES["numba"] = "numba"

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
    with pytest.warns(FromNameWarning):
        return CameraGeometry.from_name(request.param)


@pytest.fixture(scope="session")
def _global_example_event():
    """
    helper to get a single event from a MC file. Don't use this fixture
    directly, rather use `example_event`
    """
    filename = get_dataset_path("gamma_test_large.simtel.gz")

    print("******************** LOAD TEST EVENT ***********************")

    # FIXME: switch to prod5b+ file that contains effective focal length
    with SimTelEventSource(
        input_url=filename, focal_length_choice="EQUIVALENT"
    ) as reader:
        event = next(iter(reader))

    return event


@pytest.fixture(scope="session")
def subarray_prod5_paranal(prod5_gamma_simtel_path):
    return SubarrayDescription.read(prod5_gamma_simtel_path)


@pytest.fixture(scope="session")
def subarray_prod3_paranal():
    return SubarrayDescription.read(
        "dataset://gamma_test_large.simtel.gz",
        focal_length_choice="EQUIVALENT",
    )


@pytest.fixture(scope="session")
def example_subarray(subarray_prod3_paranal):
    """
    Subarray corresponding to the example event
    """
    return subarray_prod3_paranal


@pytest.fixture(scope="function")
def example_event(_global_example_event):
    """
    Use this fixture anywhere you need a test event read from a MC file. For
    example:

    .. code-block::
        def test_my_thing(example_event):
            assert len(example_event.r0.tel) > 0

    """
    return deepcopy(_global_example_event)


@pytest.fixture(scope="session")
def _subarray_and_event_gamma_off_axis_500_gev():
    from ctapipe.calib import CameraCalibrator
    from ctapipe.image import ImageProcessor

    path = get_dataset_path("lst_prod3_calibration_and_mcphotons.simtel.zst")

    with SimTelEventSource(path, focal_length_choice="EQUIVALENT") as source:
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
    return get_dataset_path("gamma_prod5.simtel.zst")


@pytest.fixture(scope="session")
def prod5_gamma_lapalma_simtel_path():
    return get_dataset_path(
        "gamma_20deg_0deg_run1___cta-prod5-lapalma_desert-2158m-LaPalma-dark_100evts.simtel.zst"
    )


@pytest.fixture(scope="session")
def prod5_proton_simtel_path():
    return get_dataset_path(
        "proton_20deg_0deg_run4___cta-prod5-paranal_desert-2147m-Paranal-dark-100evts.simtel.zst"
    )


@pytest.fixture(scope="session")
def prod5_lst(subarray_prod5_paranal):
    return subarray_prod5_paranal.tel[1]


@pytest.fixture(scope="session")
def prod3_lst(subarray_prod3_paranal):
    return subarray_prod3_paranal.tel[1]


@pytest.fixture(scope="session")
def prod5_mst_flashcam(subarray_prod5_paranal):
    return subarray_prod5_paranal.tel[5]


@pytest.fixture(scope="session")
def prod5_mst_nectarcam(subarray_prod5_paranal):
    return subarray_prod5_paranal.tel[100]


@pytest.fixture(scope="session")
def prod5_sst(subarray_prod5_paranal):
    return subarray_prod5_paranal.tel[60]


@pytest.fixture(scope="session")
def prod3_astri(subarray_prod3_paranal):
    return subarray_prod3_paranal.tel[98]


@pytest.fixture(scope="session")
def dl1_tmp_path(tmp_path_factory):
    """Temporary directory for global dl1 test data"""
    return tmp_path_factory.mktemp("dl1_")


@pytest.fixture(scope="session")
def dl2_tmp_path(tmp_path_factory):
    """Temporary directory for global dl2 test data"""
    return tmp_path_factory.mktemp("dl2_")


@pytest.fixture(scope="session")
def dl2_shower_geometry_file(dl2_tmp_path, prod5_gamma_simtel_path):
    """
    File containing both parameters and shower geometry from a gamma simulation set.
    """
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
            "--write-showers",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl2_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl2_shower_geometry_file_lapalma(dl2_tmp_path, prod5_gamma_lapalma_simtel_path):
    """
    File containing both parameters and shower geometry from a gamma simulation set.
    """
    from ctapipe.tools.process import ProcessorTool

    output = dl2_tmp_path / "gamma_lapalma.training.h5"

    # prevent running process multiple times in case of parallel tests
    with FileLock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file():
            return output

        argv = [
            f"--input={prod5_gamma_lapalma_simtel_path}",
            f"--output={output}",
            "--write-images",
            "--write-showers",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl2_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl2_proton_geometry_file(dl2_tmp_path, prod5_proton_simtel_path):
    """
    File containing both parameters and shower geometry from a gamma simulation set.
    """
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
            "--write-showers",
            "--max-events=20",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl2_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl2_merged_file(dl2_tmp_path, dl2_shower_geometry_file, dl2_proton_geometry_file):
    """
    File containing both parameters and shower geometry from a gamma simulation set.
    """
    from ctapipe.tools.merge import MergeTool

    output = dl2_tmp_path / "merged.training.h5"

    # prevent running process multiple times in case of parallel tests
    with FileLock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file():
            return output

        argv = [
            f"--output={output}",
            str(dl2_proton_geometry_file),
            str(dl2_shower_geometry_file),
        ]
        assert run_tool(MergeTool(), argv=argv, cwd=dl2_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl1_file(dl1_tmp_path, prod5_gamma_simtel_path):
    """
    DL1 file containing both images and parameters from a gamma simulation set.
    """
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
def dl1_divergent_file(dl1_tmp_path):
    from ctapipe.tools.process import ProcessorTool

    path = "dataset://gamma_divergent_LaPalma_baseline_20Zd_180Az_prod3_test.simtel.gz"
    output = dl1_tmp_path / "gamma_divergent.dl1.h5"

    # prevent running process multiple times in case of parallel tests
    with FileLock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file():
            return output

        argv = [
            f"--input={path}",
            f"--output={output}",
            "--EventSource.focal_length_choice=EQUIVALENT",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl1_camera_frame_file(dl1_tmp_path, prod5_gamma_simtel_path):
    """
    DL1 file containing both images and parameters from a gamma simulation set.
    """
    from ctapipe.tools.process import ProcessorTool

    output = dl1_tmp_path / "gamma_camera_frame.dl1.h5"

    # prevent running process multiple times in case of parallel tests
    with FileLock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file():
            return output

        argv = [
            f"--input={prod5_gamma_simtel_path}",
            f"--output={output}",
            "--camera-frame",
            "--write-images",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl2_only_file(dl2_tmp_path, prod5_gamma_simtel_path):
    """
    DL1 file containing both images and parameters from a gamma simulation set.
    """
    from ctapipe.tools.process import ProcessorTool

    output = dl2_tmp_path / "gamma_no_dl1.dl2.h5"

    # prevent running process multiple times in case of parallel tests
    with FileLock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file():
            return output

        argv = [
            f"--input={prod5_gamma_simtel_path}",
            f"--output={output}",
            "--write-showers",
            "--no-write-images",
            "--no-write-parameters",
            "--max-events=20",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl2_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl1_image_file(dl1_tmp_path, prod5_gamma_simtel_path):
    """
    DL1 file containing only images (DL1A) from a gamma simulation set.
    """
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
            "--DataWriter.write_dl1_parameters=False",
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
            "--DataWriter.Contact.name=αℓℓ the äüöß",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl1_muon_file(dl1_tmp_path):
    """
    DL1 file containing only images from a muon simulation set.
    """
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
            "--no-write-parameters",
            "--SimTelEventSource.focal_length_choice=EQUIVALENT",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl1_muon_output_file(dl1_tmp_path, dl1_muon_file):
    """
    DL1 file containing images, parameters and muon ring parameters
    from a muon simulation set.
    """
    from ctapipe.tools.process import ProcessorTool

    output = dl1_tmp_path / "muon_output.dl1.h5"
    pytest.importorskip("iminuit")

    # prevent running process multiple times in case of parallel tests
    with FileLock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file():
            return output

        argv = [
            f"--input={dl1_muon_file}",
            f"--output={output}",
            "--no-write-images",
            "--no-write-parameters",
            "--write-muon-parameters",
            "--HDF5EventSource.focal_length_choice=EQUIVALENT",
            "--max-events=30",
        ]
        assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl1_proton_file(dl1_tmp_path, prod5_proton_simtel_path):
    """
    DL1 file containing images and parameters for a prod5 proton run
    """
    from ctapipe.tools.process import ProcessorTool

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


@pytest.fixture(scope="session")
def model_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("models")


@pytest.fixture(scope="session")
def energy_regressor_path(model_tmp_path):
    from ctapipe.tools.train_energy_regressor import TrainEnergyRegressor

    out_file = model_tmp_path / "energy.pkl"

    with FileLock(out_file.with_suffix(out_file.suffix + ".lock")):
        if out_file.is_file():
            return out_file

        tool = TrainEnergyRegressor()
        config = resource_file("train_energy_regressor.yaml")
        ret = run_tool(
            tool,
            argv=[
                "--input=dataset://gamma_diffuse_dl2_train_small.dl2.h5",
                f"--output={out_file}",
                f"--config={config}",
                "--log-level=INFO",
                "--overwrite",
            ],
        )
        assert ret == 0
        return out_file


@pytest.fixture(scope="session")
def gamma_train_clf(model_tmp_path, energy_regressor_path):
    from ctapipe.tools.apply_models import ApplyModels

    inpath = "dataset://gamma_diffuse_dl2_train_small.dl2.h5"
    outpath = model_tmp_path / "gamma_train_clf.dl2.h5"
    run_tool(
        ApplyModels(),
        argv=[
            f"--input={inpath}",
            f"--output={outpath}",
            f"--reconstructor={energy_regressor_path}",
        ],
        raises=True,
    )
    return outpath


@pytest.fixture(scope="session")
def proton_train_clf(model_tmp_path, energy_regressor_path):
    from ctapipe.tools.apply_models import ApplyModels

    inpath = "dataset://proton_dl2_train_small.dl2.h5"
    outpath = model_tmp_path / "proton_train_clf.dl2.h5"
    run_tool(
        ApplyModels(),
        argv=[
            f"--input={inpath}",
            f"--output={outpath}",
            f"--reconstructor={energy_regressor_path}",
        ],
        raises=True,
    )

    # modify obs_ids by adding a constant, this enables merging gamma and proton files
    # which is used in the merge tool tests.
    with tables.open_file(outpath, mode="r+") as f:
        for table in f.walk_nodes("/", "Table"):
            if "obs_id" in table.colnames:
                obs_id = table.col("obs_id")
                table.modify_column(colname="obs_id", column=obs_id + 1_000_000_000)
    return outpath


@pytest.fixture(scope="session")
def particle_classifier_path(model_tmp_path, gamma_train_clf, proton_train_clf):
    from ctapipe.tools.train_particle_classifier import TrainParticleClassifier

    out_file = model_tmp_path / "particle_classifier.pkl"
    with FileLock(out_file.with_suffix(out_file.suffix + ".lock")):
        if out_file.is_file():
            return out_file

        config = resource_file("train_particle_classifier.yaml")

        ret = run_tool(
            TrainParticleClassifier(),
            argv=[
                f"--signal={gamma_train_clf}",
                f"--background={proton_train_clf}",
                f"--output={out_file}",
                f"--config={config}",
                "--log-level=INFO",
                "--overwrite",
            ],
        )
        assert ret == 0
        return out_file


@pytest.fixture(scope="session")
def disp_reconstructor_path(model_tmp_path, gamma_train_clf):
    from ctapipe.tools.train_disp_reconstructor import TrainDispReconstructor

    out_file = model_tmp_path / "disp_reconstructor.pkl"
    with FileLock(out_file.with_suffix(out_file.suffix + ".lock")):
        if out_file.is_file():
            return out_file

        config = resource_file("train_disp_reconstructor.yaml")

        ret = run_tool(
            TrainDispReconstructor(),
            argv=[
                f"--input={gamma_train_clf}",
                f"--output={out_file}",
                f"--config={config}",
                "--log-level=INFO",
            ],
        )
        assert ret == 0
        return out_file


@pytest.fixture(scope="session")
def reference_location():
    """a dummy EarthLocation to use for SubarrayDescriptions"""
    return EarthLocation(lon=-17 * u.deg, lat=28 * u.deg, height=2200 * u.m)


@pytest.fixture(scope="session")
def dl1_mon_pointing_file(dl1_file, dl1_tmp_path):
    from ctapipe.instrument import SubarrayDescription
    from ctapipe.io import read_table, write_table

    path = dl1_tmp_path / "dl1_mon_ponting.dl1.h5"
    shutil.copy(dl1_file, path)

    events = read_table(path, "/dl1/event/subarray/trigger")
    subarray = SubarrayDescription.from_hdf(path)

    # create some dummy monitoring data
    time = events["time"]
    start, stop = time[[0, -1]]
    duration = (stop - start).to_value(u.s)

    # start a bit before, go a bit longer
    dt = np.arange(-1, duration + 2, 1) * u.s
    time_mon = start + dt

    alt = (69 + 2 * dt / dt[-1]) * u.deg
    az = (180 + 5 * dt / dt[-1]) * u.deg

    table = Table({"time": time_mon, "azimuth": az, "altitude": alt})

    for tel_id in subarray.tel:
        write_table(table, path, f"/dl0/monitoring/telescope/pointing/tel_{tel_id:03d}")

    # remove static pointing table
    with tables.open_file(path, "r+") as f:
        f.remove_node("/configuration/telescope/pointing", recursive=True)

    return path


@pytest.fixture
def provenance(monkeypatch):
    from ctapipe.core import Provenance

    # the singleton nature of Provenance messes with
    # the order-independence of the tests asserting
    # the provenance contains the correct information
    # so we monkeypatch back to an empty state here
    prov = Provenance()
    monkeypatch.setattr(prov, "_activities", [])
    monkeypatch.setattr(prov, "_finished_activities", [])
    return prov
