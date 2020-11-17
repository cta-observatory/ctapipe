from ctapipe.utils import get_dataset_path
from ctapipe.io import DataLevel
from ctapipe.io.dl1eventsource import DL1EventSource
from ctapipe.io import EventSource
import astropy.units as u
import subprocess
import numpy as np
import tempfile
import pytest

d = tempfile.TemporaryDirectory()


@pytest.fixture(scope="module")
def dl1_file():
    simtel_path = get_dataset_path("gamma_test_large.simtel.gz")
    command = f"ctapipe-stage1 --input {simtel_path} --output {d.name}/testfile.dl1.h5 --write-parameters --write-images --max-events 20 --allowed-tels=[1,2,3]"
    subprocess.call(command.split(), stdout=subprocess.PIPE)
    return f"{d.name}/testfile.dl1.h5"


def test_is_compatible(dl1_file):
    simtel_path = get_dataset_path("gamma_test.simtel.gz")
    assert not DL1EventSource.is_compatible(simtel_path)
    assert DL1EventSource.is_compatible(dl1_file)
    with EventSource(input_url=dl1_file) as source:
        assert isinstance(source, DL1EventSource)


def test_metadata(dl1_file):
    with DL1EventSource(input_url=dl1_file) as source:
        assert source.is_simulation
        assert source.datalevels == (DataLevel.DL1_IMAGES, DataLevel.DL1_PARAMETERS)
        assert list(source.obs_ids) == [7514]
        assert source.mc_headers[7514].corsika_version == 6990


def test_subarray(dl1_file):
    with DL1EventSource(input_url=dl1_file) as source:
        assert source.subarray.telescope_types
        assert source.subarray.camera_types
        assert source.subarray.optics_types


def test_max_events(dl1_file):
    max_events = 5
    with DL1EventSource(input_url=dl1_file, max_events=max_events) as source:
        count = 0
        assert source.max_events == max_events  # stop iterating after max_events
        assert len(source) == 20  # total events in file
        for _ in source:
            count += 1
        assert count == max_events


def test_allowed_tels(dl1_file):
    allowed_tels = {1, 2}
    with DL1EventSource(input_url=dl1_file, allowed_tels=allowed_tels) as source:
        assert source.allowed_tels == allowed_tels
        for event in source:
            for tel in event.dl1.tel:
                assert tel in allowed_tels


def test_simulation_info(dl1_file):
    with DL1EventSource(input_url=dl1_file) as source:
        for event in source:
            assert np.isfinite(event.simulation.shower.energy)
            # the currently used file does not include true dl1 information
            # this is skipped for that reason
            for tel in event.simulation.tel:
                assert tel in event.simulation.tel
                assert event.simulation.tel[tel].true_image.any()
                assert event.simulation.tel[tel].true_parameters.hillas.x != np.nan


def test_dl1_data(dl1_file):
    with DL1EventSource(input_url=dl1_file) as source:
        for event in source:
            for tel in event.dl1.tel:
                assert event.dl1.tel[tel].image.any()
                assert event.dl1.tel[tel].parameters.hillas.x != np.nan


def test_pointing(dl1_file):
    with DL1EventSource(input_url=dl1_file) as source:
        for event in source:
            assert np.isclose(event.pointing.array_azimuth.to_value(u.deg), 0)
            assert np.isclose(event.pointing.array_altitude.to_value(u.deg), 70)
            assert event.pointing.tel
            for tel in event.pointing.tel:
                assert np.isclose(event.pointing.tel[tel].azimuth.to_value(u.deg), 0)
                assert np.isclose(event.pointing.tel[tel].altitude.to_value(u.deg), 70)
