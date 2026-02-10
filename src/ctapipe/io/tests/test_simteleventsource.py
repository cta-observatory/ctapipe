"""tests of SimTelEventSource"""

# pylint: disable=import-outside-toplevel
import copy
from itertools import zip_longest
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle, EarthLocation, Latitude
from astropy.time import Time
from traitlets.config import Config

from ctapipe.calib.camera.gainselection import ThresholdGainSelector
from ctapipe.instrument.camera.geometry import UnknownPixelShapeWarning
from ctapipe.instrument.optics import ReflectorShape
from ctapipe.io import DataLevel
from ctapipe.io.simteleventsource import (
    AtmosphereProfileKind,
    SimTelEventSource,
    apply_gain_selection,
    apply_simtel_r1_calibration,
    read_atmosphere_profile_from_simtel,
)
from ctapipe.utils import get_dataset_path

gamma_test_large_path = get_dataset_path("gamma_test_large.simtel.gz")
gamma_test_path = get_dataset_path("gamma_test.simtel.gz")
calib_events_path = get_dataset_path("lst_prod3_calibration_and_mcphotons.simtel.zst")
prod5b_path = get_dataset_path(
    "gamma_20deg_0deg_run2___cta-prod5-paranal_desert-2147m-Paranal-dark_cone10-100evts.simtel.zst"
)


def test_positional_input():
    source = SimTelEventSource(prod5b_path)
    assert source.input_url == Path(prod5b_path)


def test_simtel_event_source_on_gamma_test_one_event():
    assert SimTelEventSource.is_compatible(gamma_test_large_path)

    with SimTelEventSource(
        input_url=gamma_test_large_path,
        back_seekable=True,
        focal_length_choice="EQUIVALENT",
    ) as reader:
        assert not reader.is_stream

        for event in reader:
            if event.count > 1:
                break

        with pytest.warns(UserWarning):
            for event in reader:
                # Check generator has restarted from beginning
                assert event.count == 0
                break


def test_that_event_is_not_modified_after_loop():
    dataset = prod5b_path
    with SimTelEventSource(input_url=dataset, max_events=2) as source:
        for event in source:
            last_event = copy.deepcopy(event)

        # now `event` should be identical with the deepcopy of itself from
        # inside the loop.
        # Unfortunately this does not work:
        #      assert last_event == event
        # So for the moment we just compare event ids
        assert event.index.event_id == last_event.index.event_id


def test_additional_meta_data_from_simulation_config():
    with SimTelEventSource(
        input_url=gamma_test_large_path,
        focal_length_choice="EQUIVALENT",
    ) as reader:
        pass

    # There should be only one observation
    assert len(reader.obs_ids) == 1
    simulation_config = reader.simulation_config[reader.obs_ids[0]]

    assert simulation_config.corsika_version == 6990
    assert simulation_config.spectral_index == -2.0
    assert simulation_config.shower_reuse == 20
    assert simulation_config.core_pos_mode == 1
    assert simulation_config.diffuse == 1
    assert simulation_config.atmosphere == 26

    # value read by hand from input card
    name_expectation = {
        "energy_range_min": u.Quantity(3.0e-03, u.TeV),
        "energy_range_max": u.Quantity(3.3e02, u.TeV),
        "prod_site_B_total": u.Quantity(23.11772346496582, u.uT),
        "prod_site_B_declination": Angle(0.0 * u.rad),
        "prod_site_B_inclination": Angle(-0.39641156792640686 * u.rad),
        "prod_site_alt": 2150.0 * u.m,
        "max_scatter_range": 3000.0 * u.m,
        "min_az": 0.0 * u.rad,
        "min_alt": 1.2217305 * u.rad,
        "max_viewcone_radius": 10.0 * u.deg,
        "corsika_wlen_min": 240 * u.nm,
    }

    for name, expectation in name_expectation.items():
        value = getattr(simulation_config, name)

        assert value.unit == expectation.unit
        assert np.isclose(
            value.to_value(expectation.unit), expectation.to_value(expectation.unit)
        )


def test_properties():
    source = SimTelEventSource(
        input_url=gamma_test_large_path,
        focal_length_choice="EQUIVALENT",
    )

    assert source.is_simulation
    assert source.datalevels == (DataLevel.R0, DataLevel.R1)
    assert source.obs_ids == [7514]
    assert source.simulation_config[7514].corsika_version == 6990
    assert isinstance(source.subarray.reference_location, EarthLocation)


def test_gamma_file_prod2():
    dataset = gamma_test_path

    with pytest.warns(UnknownPixelShapeWarning):
        with SimTelEventSource(
            input_url=dataset,
            focal_length_choice="EQUIVALENT",
        ) as source:
            assert source.is_compatible(dataset)
            assert source.is_stream  # using gzip subprocess makes it a stream

            for event in source:
                if event.count == 0:
                    assert event.r0.tel.keys() == {38, 47}
                elif event.count == 1:
                    assert event.r0.tel.keys() == {11, 21, 24, 26, 61, 63, 118, 119}
                else:
                    break

            # check also that for old files with no reference_locatino that we get back
            # Null Island at the right height:
            assert source.subarray.reference_location.geodetic.height > 100 * u.m
            assert np.isclose(
                source.subarray.reference_location.geodetic.lat, 0 * u.deg
            )
            assert np.isclose(
                source.subarray.reference_location.geodetic.lon, 0 * u.deg
            )


def test_max_events():
    max_events = 5
    with SimTelEventSource(
        input_url=gamma_test_large_path,
        max_events=max_events,
        focal_length_choice="EQUIVALENT",
    ) as reader:
        count = 0
        for _ in reader:
            count += 1
        assert count == max_events


def test_pointing():
    with SimTelEventSource(
        input_url=gamma_test_large_path,
        max_events=3,
        focal_length_choice="EQUIVALENT",
    ) as reader:
        for e in reader:
            assert np.isclose(e.monitoring.pointing.array_altitude.to_value(u.deg), 70)
            assert np.isclose(e.monitoring.pointing.array_azimuth.to_value(u.deg), 0)
            assert np.isnan(e.monitoring.pointing.array_ra)
            assert np.isnan(e.monitoring.pointing.array_dec)

            # normal run, all telescopes point to the array direction
            for tel_id in e.monitoring.tel.keys():
                assert u.isclose(
                    e.monitoring.pointing.array_azimuth,
                    e.monitoring.tel[tel_id].pointing.azimuth,
                )
                assert u.isclose(
                    e.monitoring.pointing.array_altitude,
                    e.monitoring.tel[tel_id].pointing.altitude,
                )


def test_allowed_telescopes():
    # test that the allowed_tels mask works:
    allowed_tels = {3, 4}
    with SimTelEventSource(
        input_url=gamma_test_large_path,
        allowed_tels=allowed_tels,
        max_events=5,
        focal_length_choice="EQUIVALENT",
    ) as reader:
        assert not allowed_tels.symmetric_difference(reader.subarray.tel_ids)
        for event in reader:
            assert set(event.r0.tel).issubset(allowed_tels)
            assert set(event.r1.tel).issubset(allowed_tels)
            assert set(event.dl0.tel).issubset(allowed_tels)
            assert set(event.trigger.tels_with_trigger).issubset(allowed_tels)
            assert set(event.monitoring.tel).issubset(allowed_tels)


def test_calibration_events():
    from ctapipe.containers import EventType

    # this test file as two of each of these types
    expected_types = [
        EventType.DARK_PEDESTAL,
        EventType.DARK_PEDESTAL,
        EventType.SKY_PEDESTAL,
        EventType.SKY_PEDESTAL,
        EventType.SINGLE_PE,
        EventType.SINGLE_PE,
        EventType.FLATFIELD,
        EventType.FLATFIELD,
        EventType.SUBARRAY,
        EventType.SUBARRAY,
    ]

    expected_ids = [-1, -2, -3, -4, -5, -6, -7, -8, 100, 200]
    with SimTelEventSource(
        input_url=calib_events_path,
        skip_calibration_events=False,
        focal_length_choice="EQUIVALENT",
    ) as reader:
        for event, expected_type, expected_id in zip_longest(
            reader, expected_types, expected_ids
        ):
            assert event.trigger.event_type is expected_type
            assert event.index.event_id == expected_id


def test_skip_r1_calibration():
    n_expected = 5
    with SimTelEventSource(
        input_url=calib_events_path,
        max_events=n_expected,
        skip_calibration_events=False,
        skip_r1_calibration=True,
        focal_length_choice="EQUIVALENT",
    ) as reader:
        n_processed = 0
        for event in reader:
            n_processed += 1
            np.testing.assert_(
                np.issubdtype(event.r0.tel[1].waveform.dtype, np.uint16),
                f"R0 waveforms dtype is {event.r0.tel[1].waveform.dtype}, expected uint16.",
            )
            np.testing.assert_(
                np.issubdtype(event.r1.tel[1].waveform.dtype, np.float32),
                f"R1 waveforms dtype is {event.r1.tel[1].waveform.dtype}, expected float32.",
            )
            np.testing.assert_array_equal(
                event.r0.tel[1].waveform,
                event.r1.tel[1].waveform,
                err_msg="R0 and R1 waveforms do not match after skipping the simtel calibration.",
            )
    assert n_processed == n_expected


def test_skip_r1_calibration_with_gain_selection():
    n_expected = 1
    with SimTelEventSource(
        input_url=calib_events_path,
        max_events=n_expected,
        skip_calibration_events=False,
        select_gain=True,
        focal_length_choice="EQUIVALENT",
    ) as reader:
        n_processed = 0
        for event in reader:
            n_processed += 1
            # R0 should have 2 channels (high gain and low gain)
            assert event.r0.tel[1].waveform.ndim == 3
            assert event.r0.tel[1].waveform.shape[0] == 2, (
                f"R0 waveforms should have 2 channels, got {event.r0.tel[1].waveform.shape[0]}"
            )
            # R1 should have 1 channel after gain selection
            assert event.r1.tel[1].waveform.ndim == 3
            assert event.r1.tel[1].waveform.shape[0] == 1, (
                f"R1 waveforms should have 1 channel after gain selection, got {event.r1.tel[1].waveform.shape[0]}"
            )
            # Check that selected_gain_channel exists and has correct shape
            assert event.r1.tel[1].selected_gain_channel is not None
            assert (
                event.r1.tel[1].selected_gain_channel.shape[0]
                == event.r1.tel[1].waveform.shape[1]
            ), "selected_gain_channel should have one entry per pixel"
    assert n_processed == n_expected


def test_time_shift():
    source = SimTelEventSource(
        input_url=calib_events_path,
        focal_length_choice="EQUIVALENT",
    )
    for event in source:
        for tel_id in event.trigger.tel.keys():
            # test time shift is applied correctly
            assert event.monitoring.tel[tel_id].camera.coefficients is not None


def test_trigger_times():
    source = SimTelEventSource(
        input_url=calib_events_path,
        focal_length_choice="EQUIVALENT",
    )
    t0 = Time("2020-05-06T15:30:00")
    t1 = Time("2020-05-06T15:40:00")

    for event in source:
        assert t0 <= event.trigger.time <= t1
        for tel_id, trigger in event.trigger.tel.items():
            # test single telescope events triggered within 50 ns
            assert 0 <= (trigger.time - event.trigger.time).to_value(u.ns) <= 50


def test_true_image():
    with SimTelEventSource(
        input_url=calib_events_path,
        focal_length_choice="EQUIVALENT",
    ) as reader:
        for event in reader:
            for tel in event.simulation.tel.values():
                assert np.count_nonzero(tel.true_image) > 0


def test_instrument():
    """Test if same telescope types share a single instance of CameraGeometry"""
    source = SimTelEventSource(
        input_url=gamma_test_large_path,
        focal_length_choice="EQUIVALENT",
    )
    subarray = source.subarray
    assert subarray.tel[1].optics.n_mirrors == 1


def test_apply_simtel_r1_calibration_1_channel():
    n_channels = 1
    n_pixels = 2048
    n_samples = 128

    r0_waveforms = np.zeros((n_channels, n_pixels, n_samples))
    pedestal = np.full((n_channels, n_pixels), 20)
    dc_to_pe = np.full((n_channels, n_pixels), 0.5)

    gain_selector = ThresholdGainSelector(threshold=90)
    r1_waveforms = apply_simtel_r1_calibration(r0_waveforms, pedestal, dc_to_pe)
    r1_waveforms, selected_gain_channel = apply_gain_selection(
        r1_waveforms, gain_selector
    )

    assert (selected_gain_channel == 0).all()
    assert r1_waveforms.ndim == 3
    assert r1_waveforms.shape == (1, n_pixels, n_samples)

    ped = pedestal
    assert r1_waveforms[0, 0, 0] == (r0_waveforms[0, 0, 0] - ped[0, 0]) * dc_to_pe[0, 0]
    assert r1_waveforms[0, 1, 0] == (r0_waveforms[0, 1, 0] - ped[0, 1]) * dc_to_pe[0, 1]


def test_apply_simtel_r1_calibration_2_channel():
    n_channels = 2
    n_pixels = 2048
    n_samples = 128

    r0_waveforms = np.zeros((n_channels, n_pixels, n_samples))
    r0_waveforms[0, 0, :] = 100
    r0_waveforms[1, :, :] = 1

    pedestal = np.zeros((n_channels, n_pixels))
    pedestal[0] = 90
    pedestal[1] = 0.9

    dc_to_pe = np.zeros((n_channels, n_pixels))
    dc_to_pe[0] = 0.01
    dc_to_pe[1] = 0.1

    gain_selector = ThresholdGainSelector(threshold=90)
    r1_waveforms = apply_simtel_r1_calibration(r0_waveforms, pedestal, dc_to_pe)
    r1_waveforms, selected_gain_channel = apply_gain_selection(
        r1_waveforms, gain_selector
    )

    assert selected_gain_channel[0] == 0
    assert (selected_gain_channel[np.arange(1, 2048)] == 0).all()
    assert r1_waveforms.ndim == 3
    assert r1_waveforms.shape == (1, n_pixels, n_samples)

    ped = pedestal
    assert r1_waveforms[0, 0, 0] == (r0_waveforms[0, 0, 0] - ped[0, 0]) * dc_to_pe[0, 0]
    assert r1_waveforms[0, 1, 0] == (r0_waveforms[0, 1, 0] - ped[0, 1]) * dc_to_pe[0, 1]


def test_focal_length_choice():
    # this file does not contain the effective focal length
    with pytest.raises(RuntimeError):
        SimTelEventSource(gamma_test_large_path)

    with pytest.raises(RuntimeError):
        SimTelEventSource(gamma_test_large_path, focal_length_choice="EFFECTIVE")

    s = SimTelEventSource(gamma_test_large_path, focal_length_choice="EQUIVALENT")
    tel = s.subarray.tel[1]
    assert tel.camera.geometry.frame.focal_length == 28 * u.m
    assert u.isclose(tel.optics.equivalent_focal_length, 28.0 * u.m, atol=0.05 * u.m)
    assert np.isnan(tel.optics.effective_focal_length)

    # this file does
    s = SimTelEventSource(prod5b_path, focal_length_choice="EFFECTIVE")
    assert u.isclose(
        s.subarray.tel[1].optics.equivalent_focal_length, 28.0 * u.m, atol=0.05 * u.m
    )
    assert u.isclose(
        s.subarray.tel[1].optics.effective_focal_length, 29.3 * u.m, atol=0.05 * u.m
    )
    assert u.isclose(
        s.subarray.tel[1].camera.geometry.frame.focal_length,
        29.3 * u.m,
        atol=0.05 * u.m,
    )

    s = SimTelEventSource(prod5b_path, focal_length_choice="EQUIVALENT")
    assert u.isclose(
        s.subarray.tel[1].optics.equivalent_focal_length, 28.0 * u.m, atol=0.05 * u.m
    )
    # check guessing of the name is not affected by focal length choice
    assert str(s.subarray.tel[1]) == "LST_LST_LSTCam"

    s = SimTelEventSource(prod5b_path, focal_length_choice="EQUIVALENT")
    assert u.isclose(
        s.subarray.tel[1].optics.equivalent_focal_length, 28.0 * u.m, atol=0.05 * u.m
    )
    assert u.isclose(
        s.subarray.tel[1].optics.effective_focal_length, 29.3 * u.m, atol=0.05 * u.m
    )
    assert u.isclose(
        s.subarray.tel[1].camera.geometry.frame.focal_length,
        28.0 * u.m,
        atol=0.05 * u.m,
    )
    # check guessing of the name is not affected by focal length choice
    assert str(s.subarray.tel[1]) == "LST_LST_LSTCam"


def test_only_config():
    config = Config()
    config.SimTelEventSource.input_url = prod5b_path

    s = SimTelEventSource(config=config)
    assert s.input_url == Path(prod5b_path).absolute()


def test_calibscale_and_calibshift(prod5_gamma_simtel_path):
    with SimTelEventSource(input_url=prod5_gamma_simtel_path, max_events=1) as source:
        event = next(iter(source))

    # make sure we actually have data
    assert len(event.r1.tel) > 0

    calib_scale = 2.0

    with SimTelEventSource(
        input_url=prod5_gamma_simtel_path, max_events=1, calib_scale=calib_scale
    ) as source:
        event_scaled = next(iter(source))

    for tel_id, r1 in event.r1.tel.items():
        np.testing.assert_allclose(
            r1.waveform[0],
            event_scaled.r1.tel[tel_id].waveform[0] / calib_scale,
            rtol=0.1,
        )

    calib_shift = 2.0  # p.e.

    with SimTelEventSource(
        input_url=prod5_gamma_simtel_path, max_events=1, calib_shift=calib_shift
    ) as source:
        event_shifted = next(iter(source))

    for tel_id, r1 in event.r1.tel.items():
        np.testing.assert_allclose(
            r1.waveform[0],
            event_shifted.r1.tel[tel_id].waveform[0] - calib_shift,
            rtol=0.1,
        )


def test_true_image_sum():
    # this file does not contain true pe info
    with SimTelEventSource(
        gamma_test_large_path,
        focal_length_choice="EQUIVALENT",
    ) as s:
        e = next(iter(s))
        assert np.all(np.isnan(sim.true_image_sum) for sim in e.simulation.tel.values())

    with SimTelEventSource(
        calib_events_path,
        focal_length_choice="EQUIVALENT",
    ) as s:
        e = next(iter(s))

        true_image_sums = {}
        for tel_id, sim_camera in e.simulation.tel.items():
            # since the test file contains both sums and individual pixel values
            # we can compare.
            assert sim_camera.true_image_sum == sim_camera.true_image.sum()
            true_image_sums[tel_id] = sim_camera.true_image_sum

    # check it also works with allowed_tels, since the values
    # are stored in a flat array in simtel
    with SimTelEventSource(
        calib_events_path,
        allowed_tels={2, 3},
        focal_length_choice="EQUIVALENT",
    ) as s:
        e = next(iter(s))
        assert e.simulation.tel[2].true_image_sum == true_image_sums[2]
        assert e.simulation.tel[3].true_image_sum == true_image_sums[3]


def test_extracted_calibevents():
    with SimTelEventSource(
        "dataset://extracted_pedestals.simtel.zst",
        skip_calibration_events=False,
    ) as s:
        i = 0
        for e in s:
            i = e.count
            # these events are simulated but do not have shower information
            assert e.simulation is not None
            assert e.simulation.shower is None
        assert i == 4


def test_simtel_metadata(monkeypatch):
    from ctapipe.instrument import guess

    # prod6 is the first prod to use the metadata system
    path = "dataset://gamma_prod6_preliminary.simtel.zst"

    with monkeypatch.context() as m:
        # remove all guessing keys so we cannot use guessing
        m.setattr(guess, "LOOKUP_TREE", {})

        with SimTelEventSource(path) as source:
            subarray = source.subarray

    assert subarray.name == "Paranal-prod6"
    assert subarray.tel[1].camera.name == "LSTcam"
    assert subarray.tel[1].optics.name == "LST"
    assert subarray.tel[1].optics.reflector_shape is ReflectorShape.PARABOLIC

    assert subarray.tel[5].camera.name == "FlashCam"
    assert subarray.tel[5].optics.name == "MST"
    assert subarray.tel[5].optics.reflector_shape is ReflectorShape.HYBRID

    tel = subarray.tel[50]
    assert tel.camera.name == "SST-Camera"
    assert tel.optics.name == "SST"
    assert tel.optics.reflector_shape is ReflectorShape.SCHWARZSCHILD_COUDER


def test_simtel_no_metadata(monkeypatch):
    from ctapipe.instrument import guess

    # prod5 was before the metadata system was introduced
    path = "dataset://gamma_prod5.simtel.zst"

    # this will use the guessing system
    with SimTelEventSource(path) as source:
        subarray = source.subarray

    assert subarray.name == "MonteCarloArray"
    assert subarray.tel[1].camera.name == "LSTCam"
    assert subarray.tel[1].optics.name == "LST"
    assert subarray.tel[1].optics.reflector_shape is ReflectorShape.PARABOLIC

    assert subarray.tel[5].camera.name == "FlashCam"
    assert subarray.tel[5].optics.name == "MST"
    assert subarray.tel[5].optics.reflector_shape is ReflectorShape.HYBRID

    tel = subarray.tel[50]
    assert tel.camera.name == "CHEC"
    assert tel.optics.name == "ASTRI"
    assert tel.optics.reflector_shape is ReflectorShape.SCHWARZSCHILD_COUDER

    # check we get all unknown telescopes if we remove the guessing keys
    with monkeypatch.context() as m:
        # remove all guessing keys so we cannot use guessing
        m.setattr(guess, "LOOKUP_TREE", {})

        with SimTelEventSource(path) as source:
            subarray = source.subarray

        assert all([t.camera.name.startswith("UNKNOWN") for t in subarray.tel.values()])
        assert all([t.optics.name.startswith("UNKNOWN") for t in subarray.tel.values()])


def test_load_atmosphere_from_simtel(prod5_gamma_simtel_path):
    """
    Load atmosphere from a SimTelEventSource
    """
    from ctapipe.atmosphere import (
        FiveLayerAtmosphereDensityProfile,
        TableAtmosphereDensityProfile,
    )

    profile = read_atmosphere_profile_from_simtel(
        prod5_gamma_simtel_path, kind=AtmosphereProfileKind.AUTO
    )
    assert isinstance(profile, TableAtmosphereDensityProfile)

    profile = read_atmosphere_profile_from_simtel(
        prod5_gamma_simtel_path, kind=AtmosphereProfileKind.TABLE
    )
    assert isinstance(profile, TableAtmosphereDensityProfile)

    profile = read_atmosphere_profile_from_simtel(
        prod5_gamma_simtel_path, kind=AtmosphereProfileKind.FIVELAYER
    )
    assert isinstance(profile, FiveLayerAtmosphereDensityProfile)

    # old simtel files don't have the profile in them, so a null list should be
    # returned
    simtel_path_old = get_dataset_path("gamma_test_large.simtel.gz")
    profile = read_atmosphere_profile_from_simtel(simtel_path_old)
    assert not profile


def test_atmosphere_profile(prod5_gamma_simtel_path):
    """check that for a file with a profile in it that we get it back"""
    from ctapipe.atmosphere import AtmosphereDensityProfile

    with SimTelEventSource(prod5_gamma_simtel_path) as source:
        assert isinstance(source.atmosphere_density_profile, AtmosphereDensityProfile)


@pytest.mark.parametrize("sign", (-1, 1))
def test_float32_pihalf(sign):
    float32_pihalf = np.float32(sign * np.pi / 2)
    tracking_position = {"azimuth_raw": 0, "altitude_raw": float32_pihalf}
    pointing = SimTelEventSource._fill_event_pointing(tracking_position)
    # check that we changed the value to float64 pi/2 to avoid astropy error
    assert pointing.altitude.value == sign * np.pi / 2
    # check we can create a Latitude:
    Latitude(pointing.altitude.value, u.rad)

    event = {
        "mc_shower": {
            "energy": 1.0,
            "altitude": float32_pihalf,
            "azimuth": 0.0,
            "h_first_int": 20e3,
            "xmax": 350,
            "primary_id": 1,
        },
        "mc_event": {
            "xcore": 0.0,
            "ycore": 50.0,
        },
    }
    shower = SimTelEventSource._fill_simulated_event_information(event)
    assert shower.alt.value == sign * np.pi / 2
    # check we cana create a Latitude:
    Latitude(shower.alt.value, u.rad)


def test_starting_grammage():
    path = "dataset://lst_muons.simtel.zst"

    with SimTelEventSource(path, focal_length_choice="EQUIVALENT") as source:
        e = next(iter(source))
        assert e.simulation.shower.starting_grammage == 580 * u.g / u.cm**2


@pytest.mark.parametrize("override_obs_id,expected_obs_id", [(None, 1), (5, 5)])
def test_override_obs_id(override_obs_id, expected_obs_id, prod5_gamma_simtel_path):
    """Test for the override_obs_id option"""
    original_run_number = 1

    with SimTelEventSource(
        prod5_gamma_simtel_path, override_obs_id=override_obs_id
    ) as s:
        assert s.obs_id == expected_obs_id
        assert s.obs_ids == [expected_obs_id]

        assert s.simulation_config.keys() == {expected_obs_id}
        assert s.observation_blocks.keys() == {expected_obs_id}
        assert s.scheduling_blocks.keys() == {expected_obs_id}

        # this should always be the original run number
        assert s.simulation_config[s.obs_id].run_number == original_run_number

        for e in s:
            assert e.index.obs_id == expected_obs_id


def test_shower_distribution(prod5_gamma_simtel_path):
    with SimTelEventSource(prod5_gamma_simtel_path) as source:
        with pytest.warns(match="eventio file has no"):
            assert source.simulated_shower_distributions == {}

        for e in source:
            pass

        distributions = source.simulated_shower_distributions
        assert len(distributions) == 1
        distribution = distributions[source.obs_id]
        assert distribution.n_entries == 1000


def test_provenance(provenance, prod5_gamma_simtel_path):
    provenance.start_activity("test_simteleventsource")

    with SimTelEventSource(prod5_gamma_simtel_path):
        pass

    inputs = provenance.current_activity.input
    assert len(inputs) == 1
    assert inputs[0]["url"] == str(prod5_gamma_simtel_path)
    assert inputs[0]["reference_meta"] is None


def test_prod6_issues():
    """Test behavior of source on file from prod6, see issues #2344 and #2660"""
    input_url = "dataset://prod6_issues.simtel.zst"

    events_checked_trigger = set()
    events_checked_image = set()

    # events with two telescope events but only one in stereo trigger in simtel
    strange_trigger_events = {
        1548602: 3,
        2247909: 32,
        3974908: 2,
        4839806: 1,
    }
    missing_true_images = {1664106: 32}

    with SimTelEventSource(input_url) as source:
        for e in source:
            event_id = e.index.event_id
            if event_id in strange_trigger_events:
                expected_tel_id = strange_trigger_events[event_id]
                np.testing.assert_equal(e.trigger.tels_with_trigger, [expected_tel_id])
                assert e.trigger.tel.keys() == {expected_tel_id}
                assert e.r1.tel.keys() == {expected_tel_id}
                events_checked_trigger.add(event_id)

            if event_id in missing_true_images:
                tel_id = missing_true_images[event_id]
                np.testing.assert_equal(e.simulation.tel[tel_id].true_image, -1)
                events_checked_image.add(event_id)

    assert strange_trigger_events.keys() == events_checked_trigger
    assert missing_true_images.keys() == events_checked_image


@pytest.mark.parametrize(
    ("path", "expected", "kwargs"),
    [
        ("dataset://gamma_prod5.simtel.zst", True, {}),
        ("dataset://test_pe_first_missing.simtel.zst", True, {}),
        (gamma_test_large_path, False, {"focal_length_choice": "EQUIVALENT"}),
    ],
)
def test_has_true_image(path, expected, kwargs):
    with SimTelEventSource(path, **kwargs) as source:
        assert source._has_true_image == expected


def test_has_true_image_first_missing():
    """Check that we have true images even if the first event is missing it"""
    input_url = "dataset://test_pe_first_missing.simtel.zst"

    with SimTelEventSource(input_url) as source:
        n_found = 0
        for event in source:
            assert event.simulation.tel[1].true_image is not None
            n_found += 1
        assert n_found == 2
