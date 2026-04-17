from itertools import zip_longest

import astropy.units as u
import numpy as np
import pytest

from ctapipe.containers import (
    CameraHillasParametersContainer,
    CameraTimingParametersContainer,
    ObservationBlockContainer,
)
from ctapipe.io import DATA_MODEL_VERSION, DataLevel, EventSource, HDF5EventSource


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
    expected = tuple(map(int, DATA_MODEL_VERSION.removeprefix("v").split(".")))
    with HDF5EventSource(input_url=dl1_file) as source:
        assert source.is_simulation
        assert source.datamodel_version == expected
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
            assert set(event.monitoring.tel).issubset(allowed_tels)
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
            for tel_event in event.tel.values():
                assert tel_event.simulation is not None
                assert tel_event.simulation.true_image is not None
                reco_lons.append(
                    tel_event.simulation.true_parameters.hillas.fov_lon.value
                )
                reco_concentrations.append(
                    tel_event.simulation.true_parameters.concentration.core
                )
    assert not np.isnan(reco_lons).all()
    assert sum(np.isnan(reco_lons)) == sum(np.isnan(reco_concentrations))


def test_dl1_a_only_data(dl1_image_file):
    with HDF5EventSource(input_url=dl1_image_file) as source:
        assert source.datalevels == (DataLevel.DL1_IMAGES,)
        for event in source:
            for tel_event in event.tel.values():
                assert tel_event.dl1.image.any()


def test_dl1_b_only_data(dl1_parameters_file):
    reco_lons = []
    reco_concentrations = []
    with HDF5EventSource(input_url=dl1_parameters_file) as source:
        assert source.datalevels == (DataLevel.DL1_PARAMETERS,)
        for event in source:
            for tel_event in event.tel.values():
                reco_lons.append(
                    tel_event.simulation.true_parameters.hillas.fov_lon.value
                )
                reco_concentrations.append(
                    tel_event.simulation.true_parameters.concentration.core
                )
    assert not np.isnan(reco_lons).all()
    assert sum(np.isnan(reco_lons)) == sum(np.isnan(reco_concentrations))


def test_dl1_data(dl1_file):
    reco_lons = []
    reco_concentrations = []
    with HDF5EventSource(input_url=dl1_file) as source:
        for event in source:
            for tel_event in event.tel.values():
                assert tel_event.dl1.image.any()
                reco_lons.append(
                    tel_event.simulation.true_parameters.hillas.fov_lon.value
                )
                reco_concentrations.append(
                    tel_event.simulation.true_parameters.concentration.core
                )

    assert not np.isnan(reco_lons).all()
    assert sum(np.isnan(reco_lons)) == sum(np.isnan(reco_concentrations))


def test_pointing(dl1_file):
    with HDF5EventSource(input_url=dl1_file) as source:
        for event in source:
            pointing = event.monitoring.pointing
            assert np.isclose(pointing.azimuth.to_value(u.deg), 0)
            assert np.isclose(pointing.altitude.to_value(u.deg), 70)
            for tel_event in event.tel.values():
                pointing = tel_event.monitoring.pointing
                assert np.isclose(pointing.azimuth.to_value(u.deg), 0)
                assert np.isclose(pointing.altitude.to_value(u.deg), 70)


def test_pointing_divergent(dl1_divergent_file):
    path = "dataset://gamma_divergent_LaPalma_baseline_20Zd_180Az_prod3_test.simtel.gz"

    source = HDF5EventSource(
        input_url=dl1_divergent_file, focal_length_choice="EQUIVALENT"
    )
    simtel_source = EventSource(path, focal_length_choice="EQUIVALENT")

    with source, simtel_source:
        for event, simtel_event in zip_longest(source, simtel_source):
            assert event.index.event_id == simtel_event.index.event_id
            pointing = event.monitoring.pointing
            simtel_pointing = simtel_event.monitoring.pointing
            assert u.isclose(pointing.azimuth, simtel_pointing.azimuth)
            assert u.isclose(pointing.altitude, simtel_pointing.altitude)

            assert event.tel.keys() == simtel_event.tel.keys()
            for tel_id in event.tel:
                pointing = event.tel[tel_id].monitoring.pointing
                simtel_pointing = simtel_event.tel[tel_id].monitoring.pointing
                assert u.isclose(pointing.azimuth, simtel_pointing.azimuth)
                assert u.isclose(pointing.altitude, simtel_pointing.altitude)


def test_read_r1(r1_hdf5_file):
    with HDF5EventSource(input_url=r1_hdf5_file) as source:
        e = None

        assert source.datalevels == (DataLevel.R1,)

        for e in source:
            for tel_event in e.tel.values():
                assert tel_event.r1.waveform is not None

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
            assert set(e.dl0.trigger.tels_with_trigger) == e.tel.keys()
            assert len(e.dl0.trigger.tels_with_trigger) > 1

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
        assert algorithm in e.dl2.geometry
        assert e.dl2.geometry[algorithm].alt is not None
        assert e.dl2.geometry[algorithm].az is not None
        assert e.dl2.geometry[algorithm].telescopes is not None
        assert e.dl2.geometry[algorithm].prefix == algorithm

        tel_mask = e.dl2.geometry[algorithm].telescopes
        tel_ids = s.subarray.tel_mask_to_tel_ids(tel_mask)
        for tel_id in tel_ids:
            assert tel_id in e.tel
            assert algorithm in e.tel[tel_id].dl2.impact
            impact = e.tel[tel_id].dl2.impact[algorithm]
            assert impact.prefix == algorithm + "_tel_impact"
            assert impact.distance is not None


def test_dl1_camera_frame(dl1_camera_frame_file):
    with HDF5EventSource(dl1_camera_frame_file) as s:
        tel_id = None
        for e in s:
            for tel_id, tel_event in e.tel.items():
                dl1 = tel_event.dl1
                assert isinstance(
                    dl1.parameters.hillas, CameraHillasParametersContainer
                )
                assert isinstance(
                    dl1.parameters.timing, CameraTimingParametersContainer
                )
                assert dl1.parameters.hillas.intensity is not None

                sim = tel_event.simulation
                assert isinstance(
                    sim.true_parameters.hillas, CameraHillasParametersContainer
                )
                assert sim.true_parameters.hillas.intensity is not None

        assert tel_id is not None, "did not test any events"


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
            assert e.tel.keys() == set(e.dl0.trigger.tels_with_trigger)
            for tel_event in e.tel.values():
                assert u.isclose(tel_event.monitoring.pointing.altitude, 70 * u.deg)
                assert u.isclose(tel_event.monitoring.pointing.azimuth, 0 * u.deg)
            n_read += 1
    assert n_read == 5


def test_no_pointing_in_ob(tmp_path):
    from ctapipe.io import DataWriter

    test_file = "dataset://gamma_prod5.simtel.zst"

    path = tmp_path / "test_no_pointing_in_ob.h5"

    with EventSource(input_url=test_file) as source:
        # clear everything but the obs_id
        source.observation_blocks[source.obs_id] = ObservationBlockContainer(
            obs_id=source.obs_id
        )

        n_written = 0
        with DataWriter(source, output_path=path, write_r1_waveforms=True) as writer:
            for event in source:
                writer(event)
                n_written += 1

    with HDF5EventSource(path) as source:
        n_read = 0
        for e in source:
            assert np.isnan(e.monitoring.pointing.azimuth)
            assert np.isnan(e.monitoring.pointing.altitude)
            n_read += 1
        assert n_read == n_written


def test_read_dl2_tel_ml(gamma_diffuse_full_reco_file):
    algorithm = "ExtraTreesRegressor"

    with HDF5EventSource(gamma_diffuse_full_reco_file) as s:
        assert s.datalevels == (DataLevel.DL2,)

        e = next(iter(s))
        assert algorithm in e.dl2.energy
        assert e.dl2.energy[algorithm].energy is not None

        tel_mask = e.dl2.energy[algorithm].telescopes
        tel_ids = s.subarray.tel_mask_to_tel_ids(tel_mask)
        for tel_id in tel_ids:
            assert tel_id in e.tel
            tel_event = e.tel[tel_id]

            assert algorithm in tel_event.dl2.energy
            energy = tel_event.dl2.energy[algorithm]
            assert energy.prefix == algorithm + "_tel"
            assert energy.energy is not None
            assert np.isfinite(energy.energy)


def test_is_compatible_with_only_trigger(tmp_path):
    """
    Regression test for has_trigger copy-paste bug.
    """

    import tables

    filename = tmp_path / "only_trigger.h5"

    with tables.open_file(filename, mode="w") as h5:
        h5.root._v_attrs["CTA PRODUCT DATA MODEL VERSION"] = DATA_MODEL_VERSION

        h5.root._v_attrs["CTA PRODUCT DATA LEVELS"] = "R0"

        h5.create_group("/", "dl1")
        h5.create_group("/dl1", "event")
        h5.create_group("/dl1/event", "subarray")
        h5.create_group("/dl1/event/subarray", "trigger")

    assert HDF5EventSource.is_compatible(str(filename))
