import astropy.units as u
import numpy as np
from ctapipe.io import DataLevel, EventSource, HDF5EventSource
from ctapipe.utils import get_dataset_path


def test_is_compatible(dl1_file):
    simtel_path = get_dataset_path("gamma_test_large.simtel.gz")
    assert not HDF5EventSource.is_compatible(simtel_path)
    assert HDF5EventSource.is_compatible(dl1_file)
    with EventSource(input_url=dl1_file) as source:
        assert isinstance(source, HDF5EventSource)


def test_metadata(dl1_file):
    with HDF5EventSource(input_url=dl1_file) as source:
        assert source.is_simulation
        assert set(source.datalevels) == {
            DataLevel.DL1_IMAGES,
            DataLevel.DL1_PARAMETERS,
        }
        assert list(source.obs_ids) == [2]
        assert source.simulation_config[2].corsika_version == 7710


def test_subarray(dl1_file):
    with HDF5EventSource(input_url=dl1_file) as source:
        assert source.subarray.telescope_types
        assert source.subarray.camera_types
        assert source.subarray.optics_types


def test_max_events(dl1_proton_file):
    max_events = 3
    with HDF5EventSource(input_url=dl1_proton_file, max_events=max_events) as source:
        assert source.max_events == max_events  # stop iterating after max_events
        assert len(source) == 4  # total events in file
        for count, _ in enumerate(source, start=1):
            pass
        assert count == max_events


def test_allowed_tels(dl1_file):
    allowed_tels = {1, 2}
    with HDF5EventSource(input_url=dl1_file, allowed_tels=allowed_tels) as source:
        assert not allowed_tels.symmetric_difference(source.subarray.tel_ids)
        assert source.allowed_tels == allowed_tels
        for event in source:
            assert set(event.trigger.tels_with_trigger).issubset(allowed_tels)
            assert set(event.pointing.tel).issubset(allowed_tels)
            assert set(event.dl1.tel).issubset(allowed_tels)


def test_simulation_info(dl1_file):
    """
    Test that the simulated event information is plausible.
    In particular this means simulated event information is finite
    for all events and parameters calculated on the true images
    are not all nan with the same number of nans in different columns.
    """
    reco_lons = []
    reco_concentrations = []
    with HDF5EventSource(input_url=dl1_file) as source:
        for event in source:
            assert np.isfinite(event.simulation.shower.energy)
            for tel in event.simulation.tel:
                assert tel in event.simulation.tel
                assert event.simulation.tel[tel].true_image is not None
                reco_lons.append(
                    event.simulation.tel[tel].true_parameters.hillas.fov_lon.value
                )
                reco_concentrations.append(
                    event.simulation.tel[tel].true_parameters.concentration.core
                )
    assert not np.isnan(reco_lons).all()
    assert sum(np.isnan(reco_lons)) == sum(np.isnan(reco_concentrations))


def test_dl1_a_only_data(dl1_image_file):
    with HDF5EventSource(input_url=dl1_image_file) as source:
        for event in source:
            for tel in event.dl1.tel:
                assert event.dl1.tel[tel].image.any()


def test_dl1_b_only_data(dl1_parameters_file):
    reco_lons = []
    reco_concentrations = []
    with HDF5EventSource(input_url=dl1_parameters_file) as source:
        for event in source:
            for tel in event.dl1.tel:
                reco_lons.append(
                    event.simulation.tel[tel].true_parameters.hillas.fov_lon.value
                )
                reco_concentrations.append(
                    event.simulation.tel[tel].true_parameters.concentration.core
                )
    assert not np.isnan(reco_lons).all()
    assert sum(np.isnan(reco_lons)) == sum(np.isnan(reco_concentrations))


def test_dl1_data(dl1_file):
    reco_lons = []
    reco_concentrations = []
    with HDF5EventSource(input_url=dl1_file) as source:
        for event in source:
            for tel in event.dl1.tel:
                assert event.dl1.tel[tel].image.any()
                reco_lons.append(
                    event.simulation.tel[tel].true_parameters.hillas.fov_lon.value
                )
                reco_concentrations.append(
                    event.simulation.tel[tel].true_parameters.concentration.core
                )
    assert not np.isnan(reco_lons).all()
    assert sum(np.isnan(reco_lons)) == sum(np.isnan(reco_concentrations))


def test_pointing(dl1_file):
    with HDF5EventSource(input_url=dl1_file) as source:
        for event in source:
            assert np.isclose(event.pointing.array_azimuth.to_value(u.deg), 0)
            assert np.isclose(event.pointing.array_altitude.to_value(u.deg), 70)
            assert event.pointing.tel
            for tel in event.pointing.tel:
                assert np.isclose(event.pointing.tel[tel].azimuth.to_value(u.deg), 0)
                assert np.isclose(event.pointing.tel[tel].altitude.to_value(u.deg), 70)


def test_read_r1(r1_hdf5_file):
    print(r1_hdf5_file)
    with HDF5EventSource(input_url=r1_hdf5_file) as source:
        e = None

        for e in source:
            pass

        assert e is not None
        assert e.count == 4
