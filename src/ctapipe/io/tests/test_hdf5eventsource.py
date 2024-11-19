from itertools import zip_longest

import astropy.units as u
import numpy as np
import pytest

from ctapipe.containers import (
    CameraHillasParametersContainer,
    CameraTimingParametersContainer,
)
from ctapipe.io import DataLevel, EventSource, HDF5EventSource


def test_is_not_compatible(prod5_gamma_simtel_path):
    assert not HDF5EventSource.is_compatible(prod5_gamma_simtel_path)


@pytest.mark.parametrize(
    "compatible_file", ["dl1_file", "dl2_only_file", "dl1_muon_output_file"]
)
def test_is_compatible(compatible_file, request):
    old_files = {"dl1_muon_output_file"}
    focal_length = "EQUIVALENT" if compatible_file in old_files else "EFFECTIVE"
    file = request.getfixturevalue(compatible_file)
    assert HDF5EventSource.is_compatible(file)
    with EventSource(input_url=file, focal_length_choice=focal_length) as source:
        assert isinstance(source, HDF5EventSource)


def test_metadata(dl1_file):
    with HDF5EventSource(input_url=dl1_file) as source:
        assert source.is_simulation
        assert source.datamodel_version == (6, 0, 0)
        assert set(source.datalevels) == {
            DataLevel.DL1_IMAGES,
            DataLevel.DL1_PARAMETERS,
        }
        assert list(source.obs_ids) == [1]
        assert source.simulation_config[1].corsika_version == 7710


def test_subarray(dl1_file):
    with HDF5EventSource(input_url=dl1_file) as source:
        assert source.subarray.telescope_types
        assert source.subarray.camera_types
        assert source.subarray.optics_types


def test_max_events(dl1_proton_file):
    max_events = 3
    with HDF5EventSource(input_url=dl1_proton_file, max_events=max_events) as source:
        assert source.max_events == max_events  # stop iterating after max_events
        assert len(source) == 3
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
        assert source.datalevels == (DataLevel.DL1_IMAGES,)
        for event in source:
            for tel in event.dl1.tel:
                assert event.dl1.tel[tel].image.any()


def test_dl1_b_only_data(dl1_parameters_file):
    reco_lons = []
    reco_concentrations = []
    with HDF5EventSource(input_url=dl1_parameters_file) as source:
        assert source.datalevels == (DataLevel.DL1_PARAMETERS,)
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


def test_pointing_divergent(dl1_divergent_file):
    path = "dataset://gamma_divergent_LaPalma_baseline_20Zd_180Az_prod3_test.simtel.gz"

    source = HDF5EventSource(
        input_url=dl1_divergent_file, focal_length_choice="EQUIVALENT"
    )
    simtel_source = EventSource(path, focal_length_choice="EQUIVALENT")

    with source, simtel_source:
        for event, simtel_event in zip_longest(source, simtel_source):
            assert event.index.event_id == simtel_event.index.event_id
            assert u.isclose(
                event.pointing.array_azimuth,
                simtel_event.pointing.array_azimuth,
            )
            assert u.isclose(
                event.pointing.array_altitude,
                simtel_event.pointing.array_altitude,
            )
            assert event.pointing.tel.keys() == simtel_event.pointing.tel.keys()
            for tel in event.pointing.tel:
                assert u.isclose(
                    event.pointing.tel[tel].azimuth,
                    simtel_event.pointing.tel[tel].azimuth,
                )
                assert u.isclose(
                    event.pointing.tel[tel].altitude,
                    simtel_event.pointing.tel[tel].altitude,
                )


def test_read_r1(r1_hdf5_file):
    print(r1_hdf5_file)
    with HDF5EventSource(input_url=r1_hdf5_file) as source:
        e = None

        assert source.datalevels == (DataLevel.R1,)

        for e in source:
            pass

        assert e is not None
        assert e.count == 3


def test_trigger_allowed_tels(dl1_proton_file):
    with HDF5EventSource(
        input_url=dl1_proton_file, allowed_tels={1, 2, 3, 4, 5, 10}
    ) as s:
        print()
        i = 0
        for i, e in enumerate(s):
            assert e.count == i
            assert set(e.trigger.tels_with_trigger) == e.trigger.tel.keys()
            assert len(e.trigger.tels_with_trigger) > 1

        assert i == 1


def test_read_dl2(dl2_shower_geometry_file):
    algorithm = "HillasReconstructor"

    with HDF5EventSource(dl2_shower_geometry_file) as s:
        assert s.datalevels == (
            DataLevel.DL1_IMAGES,
            DataLevel.DL1_PARAMETERS,
            DataLevel.DL2,
        )

        e = next(iter(s))
        assert algorithm in e.dl2.stereo.geometry
        assert e.dl2.stereo.geometry[algorithm].alt is not None
        assert e.dl2.stereo.geometry[algorithm].az is not None
        assert e.dl2.stereo.geometry[algorithm].telescopes is not None
        assert e.dl2.stereo.geometry[algorithm].prefix == algorithm

        tel_mask = e.dl2.stereo.geometry[algorithm].telescopes
        tel_ids = s.subarray.tel_mask_to_tel_ids(tel_mask)
        for tel_id in tel_ids:
            assert tel_id in e.dl2.tel
            assert algorithm in e.dl2.tel[tel_id].impact
            impact = e.dl2.tel[tel_id].impact[algorithm]
            assert impact.prefix == algorithm + "_tel_impact"
            assert impact.distance is not None


def test_dl1_camera_frame(dl1_camera_frame_file):
    with HDF5EventSource(dl1_camera_frame_file) as s:
        tel_id = None
        for e in s:
            for tel_id, dl1 in e.dl1.tel.items():
                assert isinstance(
                    dl1.parameters.hillas, CameraHillasParametersContainer
                )
                assert isinstance(
                    dl1.parameters.timing, CameraTimingParametersContainer
                )
                assert dl1.parameters.hillas.intensity is not None

            for tel_id, sim in e.simulation.tel.items():
                assert isinstance(
                    sim.true_parameters.hillas, CameraHillasParametersContainer
                )
                assert sim.true_parameters.hillas.intensity is not None

        assert tel_id is not None, "did not test any events"


def test_interpolate_pointing(dl1_mon_pointing_file):
    from ctapipe.io import HDF5EventSource

    with HDF5EventSource(dl1_mon_pointing_file) as source:
        for e in source:
            assert set(e.pointing.tel.keys()) == set(e.trigger.tels_with_trigger)
            for pointing in e.pointing.tel.values():
                assert not np.isnan(pointing.azimuth)
                assert not np.isnan(pointing.altitude)


def test_simulated_events_distribution(dl1_file):
    with HDF5EventSource(dl1_file) as source:
        assert len(source.simulated_shower_distributions) == 1
        dist = source.simulated_shower_distributions[1]
        assert dist["n_entries"] == 1000
        assert dist["histogram"].sum() == 1000.0


def test_provenance(dl1_file, provenance):
    """Make sure that HDF5EventSource reads reference metadata and adds to provenance"""
    from ctapipe.io.metadata import _read_reference_metadata_hdf5

    provenance.start_activity("test_hdf5eventsource")
    with HDF5EventSource(input_url=dl1_file):
        pass

    inputs = provenance.current_activity.input
    assert len(inputs) == 1
    assert inputs[0]["url"] == str(dl1_file)
    meta = _read_reference_metadata_hdf5(dl1_file)
    assert inputs[0]["reference_meta"].product.id_ == meta.product.id_


def test_pointing_old_file():
    input_url = "dataset://gamma_diffuse_dl2_train_small.dl2.h5"

    n_read = 0
    with HDF5EventSource(input_url, max_events=5) as source:
        for e in source:
            assert e.pointing.tel.keys() == set(e.trigger.tels_with_trigger)
            for pointing in e.pointing.tel.values():
                assert u.isclose(pointing.altitude, 70 * u.deg)
                assert u.isclose(pointing.azimuth, 0 * u.deg)
            n_read += 1
    assert n_read == 5
