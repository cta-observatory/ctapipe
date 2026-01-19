"""
common pytest fixtures for tests in ctapipe
"""

import importlib
import importlib.util
import json
import shutil
import warnings
from copy import deepcopy
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
import tables
from astropy.coordinates import EarthLocation
from astropy.table import Column, QTable, Table, hstack, vstack
from pytest_astropy_header.display import PYTEST_HEADER_MODULES

from ctapipe.core import run_tool
from ctapipe.instrument import CameraGeometry, FromNameWarning, SubarrayDescription
from ctapipe.io import SimTelEventSource, read_table, write_table
from ctapipe.io.hdf5dataformat import (
    DL0_TEL_POINTING_GROUP,
    DL1_CAMERA_COEFFICIENTS_GROUP,
    DL1_GROUP,
    SIMULATION_GROUP,
)
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
collect_ignore = []

if importlib.util.find_spec("pyirf") is None:
    collect_ignore.append("irf")


@pytest.fixture(scope="function", params=camera_names)
def camera_geometry(request):
    with pytest.warns(FromNameWarning):
        return CameraGeometry.from_name(request.param)


def _lock_file(path: Path):
    return path.with_name(path.name + ".lock")


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
def prod6_gamma_simtel_path():
    return get_dataset_path("gamma_prod6_preliminary.simtel.zst")


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
def proton_dl2_train_small_h5():
    return get_dataset_path("proton_dl2_train_small.dl2.h5")


@pytest.fixture(scope="session")
def calibpipe_camcalib_sims_single_chunk():
    return get_dataset_path("calibpipe_camcalib_sims_single_chunk_i0.2.0.dl1.h5")


@pytest.fixture(scope="session")
def calibpipe_camcalib_obslike_same_chunks():
    return get_dataset_path("calibpipe_camcalib_obslike_same_chunks_i0.2.0.dl1.h5")


@pytest.fixture(scope="session")
def calibpipe_camcalib_obslike_different_chunks():
    return get_dataset_path("calibpipe_camcalib_obslike_different_chunks_i0.2.0.dl1.h5")


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
    with FileLock(_lock_file(output)):
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
    with FileLock(_lock_file(output)):
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
    with FileLock(_lock_file(output)):
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
    with FileLock(_lock_file(output)):
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
    with FileLock(_lock_file(output)):
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
    with FileLock(_lock_file(output)):
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
    with FileLock(_lock_file(output)):
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
    File only containing dl2 shower information.
    """
    from ctapipe.tools.process import ProcessorTool

    output = dl2_tmp_path / "gamma_no_dl1.dl2.h5"

    # prevent running process multiple times in case of parallel tests
    with FileLock(_lock_file(output)):
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
    with FileLock(_lock_file(output)):
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
    with FileLock(_lock_file(output)):
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
def dl1_tel1_file(dl1_tmp_path, prod6_gamma_simtel_path):
    """
    DL1 file containing only data for telescope 1 from a gamma simulation set.
    """
    from ctapipe.tools.process import ProcessorTool

    output = dl1_tmp_path / "gamma_tel1.dl1.h5"

    # prevent running process multiple times in case of parallel tests
    with FileLock(_lock_file(output)):
        if output.is_file():
            return output
        tel_id = 1
        argv = [
            f"--input={prod6_gamma_simtel_path}",
            f"--output={output}",
            f"--EventSource.allowed_tels={tel_id}",
            "--write-images",
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
    with FileLock(_lock_file(output)):
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
    with FileLock(_lock_file(output)):
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

    with FileLock(_lock_file(output)):
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

    with FileLock(_lock_file(out_file)):
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
    with FileLock(_lock_file(out_file)):
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
    cv_out_file = model_tmp_path / "cv_disp_reconstructor.h5"
    with FileLock(_lock_file(out_file)):
        if out_file.is_file():
            return out_file

        config = resource_file("train_disp_reconstructor.yaml")

        ret = run_tool(
            TrainDispReconstructor(),
            argv=[
                f"--input={gamma_train_clf}",
                f"--output={out_file}",
                f"--cv-output={cv_out_file}",
                f"--config={config}",
                "--log-level=INFO",
            ],
        )
        assert ret == 0
        return out_file, cv_out_file


@pytest.fixture(scope="session")
def reference_location():
    """a dummy EarthLocation to use for SubarrayDescriptions"""
    return EarthLocation(lon=-17 * u.deg, lat=28 * u.deg, height=2200 * u.m)


@pytest.fixture(scope="session")
def dl1_mon_pointing_file(calibpipe_camcalib_sims_single_chunk, dl1_tmp_path):
    path = dl1_tmp_path / "dl1_mon_pointing.dl1.h5"
    shutil.copy(calibpipe_camcalib_sims_single_chunk, path)

    tel_id = 1
    # create some dummy monitoring data
    start = read_table(
        calibpipe_camcalib_sims_single_chunk,
        f"{DL1_CAMERA_COEFFICIENTS_GROUP}/tel_{tel_id:03d}",
    )["time"][0]
    stop = start + 10 * u.s
    duration = (stop - start).to_value(u.s)

    # start a bit before, go a bit longer
    dt = np.arange(-1, duration + 2, 1) * u.s
    time_mon = start + dt

    alt = (69 + 2 * dt / dt[-1]) * u.deg
    az = (180 + 5 * dt / dt[-1]) * u.deg

    table = Table({"time": time_mon, "azimuth": az, "altitude": alt})
    write_table(table, path, f"{DL0_TEL_POINTING_GROUP}/tel_{tel_id:03d}")

    # remove static pointing table
    with tables.open_file(path, "r+") as f:
        # Remove the DL1 table
        if DL1_GROUP in f.root:
            f.remove_node(DL1_GROUP, recursive=True)

    return path


@pytest.fixture(scope="session")
def dl1_mon_pointing_file_obs(dl1_mon_pointing_file, dl1_tmp_path):
    path = dl1_tmp_path / "dl1_mon_pointing_obs.dl1.h5"
    shutil.copy(dl1_mon_pointing_file, path)

    # Remove the simulation to mimic a real observation file
    with tables.open_file(path, "r+") as f:
        data_category = "CTA PRODUCT DATA CATEGORY"
        if SIMULATION_GROUP in f.root:
            f.remove_node(SIMULATION_GROUP, recursive=True)
        if data_category in f.root._v_attrs and f.root._v_attrs[data_category] == "Sim":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", tables.NaturalNameWarning)
                f.root._v_attrs[data_category] = "Other"
    return path


@pytest.fixture(scope="session")
def dl1_merged_monitoring_file(
    dl1_tmp_path,
    dl1_tel1_file,
    dl1_mon_pointing_file,
    calibpipe_camcalib_sims_single_chunk,
):
    """
    File containing both camera and pointing monitoring data.
    """
    from ctapipe.tools.merge import MergeTool

    output = dl1_tmp_path / "dl1_merged_monitoring_file.h5"
    shutil.copy(dl1_tel1_file, output)

    # prevent running process multiple times in case of parallel tests
    with FileLock(output.with_suffix(output.suffix + ".lock")):
        argv = [
            f"--output={output}",
            str(dl1_mon_pointing_file),
            str(calibpipe_camcalib_sims_single_chunk),
            "--append",
            "--merge-strategy=monitoring-only",
        ]
        assert run_tool(MergeTool(), argv=argv, cwd=dl1_tmp_path) == 0
        return output


@pytest.fixture(scope="session")
def dl1_merged_monitoring_file_obs(dl1_merged_monitoring_file, dl1_tmp_path):
    path = dl1_tmp_path / "dl1_merged_monitoring_file_obs.dl1.h5"
    shutil.copy(dl1_merged_monitoring_file, path)

    # Remove the simulation to mimic a real observation file
    with tables.open_file(path, "r+") as f:
        data_category = "CTA PRODUCT DATA CATEGORY"
        if SIMULATION_GROUP in f.root:
            f.remove_node(SIMULATION_GROUP, recursive=True)
        if data_category in f.root._v_attrs and f.root._v_attrs[data_category] == "Sim":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", tables.NaturalNameWarning)
                f.root._v_attrs[data_category] = "Other"
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


@pytest.fixture(scope="session")
def irf_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("irf")


@pytest.fixture(scope="session")
def gamma_diffuse_full_reco_file(
    gamma_train_clf,
    particle_classifier_path,
    irf_tmp_path,
):
    """
    Energy reconstruction and geometric origin reconstruction have already been done.
    """
    from ctapipe.tools.apply_models import ApplyModels

    output_path = irf_tmp_path / "gamma_diffuse_full_reco.dl2.h5"
    run_tool(
        ApplyModels(),
        argv=[
            f"--input={gamma_train_clf}",
            f"--output={output_path}",
            f"--reconstructor={particle_classifier_path}",
            "--no-dl1-parameters",
            "--StereoMeanCombiner.weights=konrad",
        ],
        raises=True,
    )
    return output_path


@pytest.fixture(scope="session")
def proton_full_reco_file(
    proton_train_clf,
    particle_classifier_path,
    irf_tmp_path,
):
    """
    Energy reconstruction and geometric origin reconstruction have already been done.
    """
    from ctapipe.tools.apply_models import ApplyModels

    output_path = irf_tmp_path / "proton_full_reco.dl2.h5"
    run_tool(
        ApplyModels(),
        argv=[
            f"--input={proton_train_clf}",
            f"--output={output_path}",
            f"--reconstructor={particle_classifier_path}",
            "--no-dl1-parameters",
            "--StereoMeanCombiner.weights=konrad",
        ],
        raises=True,
    )
    return output_path


@pytest.fixture(scope="session")
def irf_event_loader_test_config():
    from traitlets.config import Config

    return Config(
        {
            "DL2EventPreprocessor": {
                "energy_reconstructor": "ExtraTreesRegressor",
                "geometry_reconstructor": "HillasReconstructor",
                "gammaness_classifier": "ExtraTreesClassifier",
                "DL2EventQualityQuery": {
                    "quality_criteria": [
                        (
                            "multiplicity 4",
                            "np.count_nonzero(HillasReconstructor_telescopes,axis=1) >= 4",
                        ),
                        ("valid classifier", "ExtraTreesClassifier_is_valid"),
                        ("valid geom reco", "HillasReconstructor_is_valid"),
                        ("valid energy reco", "ExtraTreesRegressor_is_valid"),
                    ],
                },
            }
        }
    )


@pytest.fixture(scope="session")
def event_loader_config_path(irf_event_loader_test_config, irf_tmp_path):
    config_path = irf_tmp_path / "event_loader_config.json"
    with config_path.open("w") as f:
        json.dump(irf_event_loader_test_config, f)

    return config_path


@pytest.fixture(scope="session")
def irf_events_table():
    from ctapipe.io import DL2EventPreprocessor

    N1 = 1000
    N2 = 100
    N = N1 + N2
    epp = DL2EventPreprocessor()
    tab = epp.make_empty_table()

    ids = ["obs_id", "event_id"]
    unitless = set(
        [colname for colname in tab.colnames if tab[colname].unit is None]
    ) - set(ids)
    bulk = set(tab.colnames) - set(ids) - set(unitless)

    id_tab = QTable(
        data=np.zeros((N, len(ids)), dtype=np.uint64),
        names=ids,
        units={c: tab[c].unit for c in ids},
    )
    bulk_tab = QTable(
        data=np.zeros((N, len(bulk))) * np.nan,
        names=bulk,
        units={c: tab[c].unit for c in bulk},
    )
    # Setting values following pyirf test in pyirf/irf/tests/test_background.py
    bulk_tab.replace_column(
        "reco_energy", np.append(np.full(N1, 1), np.full(N2, 2)) * u.TeV
    )
    bulk_tab.replace_column(
        "true_energy", np.append(np.full(N1, 0.9), np.full(N2, 2.1)) * u.TeV
    )
    bulk_tab.replace_column(
        "reco_source_fov_offset", np.append(np.full(N1, 0.1), np.full(N2, 0.05)) * u.deg
    )
    bulk_tab.replace_column(
        "true_source_fov_offset",
        np.append(np.full(N1, 0.11), np.full(N2, 0.04)) * u.deg,
    )

    for name in unitless:
        bulk_tab.add_column(
            Column(name=name, unit=tab[name].unit, data=np.zeros(N) * np.nan)
        )

    e_tab = hstack([id_tab, bulk_tab])

    ev = vstack([e_tab, tab], join_type="exact", metadata_conflicts="silent")
    return ev


@pytest.fixture(scope="function")
def test_config():
    return {
        "DL2EventLoader": {"event_reader_function": "read_telescope_events_chunked"},
        "DL2EventPreprocessor": {
            "energy_reconstructor": "ExtraTreesRegressor",
            "gammaness_classifier": "ExtraTreesClassifier",
            "columns_to_rename": {},
            "output_table_schema": [
                Column(
                    name="obs_id", dtype=np.uint64, description="Observation Block ID"
                ),
                Column(name="event_id", dtype=np.uint64, description="Array event ID"),
                Column(name="tel_id", dtype=np.uint64, description="Telescope ID"),
                Column(
                    name="ExtraTreesRegressor_tel_energy",
                    unit=u.TeV,
                    description="Reconstructed energy",
                ),
                Column(
                    name="ExtraTreesRegressor_tel_energy_uncert",
                    unit=u.TeV,
                    description="Reconstructed energy uncertainty",
                ),
            ],
            "apply_derived_columns": False,
            # "disable_column_renaming": True,
            "allow_unsupported_pointing_frames": True,
        },
        "DL2EventQualityQuery": {
            "quality_criteria": [
                ("valid reco", "ExtraTreesRegressor_tel_is_valid"),
            ]
        },
    }
