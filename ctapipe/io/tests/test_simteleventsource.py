import copy
from ctapipe.instrument.camera.geometry import UnknownPixelShapeWarning

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from itertools import zip_longest
import pytest
from astropy.time import Time
from pathlib import Path
from traitlets.config import Config


from ctapipe.calib.camera.gainselection import ThresholdGainSelector
from ctapipe.io.simteleventsource import SimTelEventSource, apply_simtel_r1_calibration
from ctapipe.utils import get_dataset_path
from ctapipe.io import DataLevel


gamma_test_large_path = get_dataset_path("gamma_test_large.simtel.gz")
gamma_test_path = get_dataset_path("gamma_test.simtel.gz")
calib_events_path = get_dataset_path("lst_prod3_calibration_and_mcphotons.simtel.zst")
prod5b_path = get_dataset_path("gamma_20deg_0deg_run2___cta-prod5-paranal_desert-2147m-Paranal-dark_cone10-100evts.simtel.zst")


def test_positional_input():
    source = SimTelEventSource(prod5b_path)
    assert source.input_url == Path(prod5b_path)


def test_simtel_event_source_on_gamma_test_one_event():
    assert SimTelEventSource.is_compatible(gamma_test_large_path)

    with SimTelEventSource(
        input_url=gamma_test_large_path, back_seekable=True,
        focal_length_choice='nominal',
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
        focal_length_choice='nominal',
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
        focal_length_choice='nominal',
    )

    assert source.is_simulation
    assert source.datalevels == (DataLevel.R0, DataLevel.R1)
    assert source.obs_ids == [7514]
    assert source.simulation_config[7514].corsika_version == 6990


def test_gamma_file_prod2():
    dataset = gamma_test_path

    with pytest.warns(UnknownPixelShapeWarning):
        with SimTelEventSource(
            input_url=dataset,
            focal_length_choice='nominal',
        ) as reader:
            assert reader.is_compatible(dataset)
            assert reader.is_stream  # using gzip subprocess makes it a stream

            for event in reader:
                if event.count == 0:
                    assert event.r0.tel.keys() == {38, 47}
                elif event.count == 1:
                    assert event.r0.tel.keys() == {11, 21, 24, 26, 61, 63, 118, 119}
                else:
                    break

    # test that max_events works:


def test_max_events():
    max_events = 5
    with SimTelEventSource(
        input_url=gamma_test_large_path,
        max_events=max_events,
        focal_length_choice='nominal',
    ) as reader:
        count = 0
        for _ in reader:
            count += 1
        assert count == max_events


def test_pointing():
    with SimTelEventSource(
        input_url=gamma_test_large_path,
        max_events=3,
        focal_length_choice='nominal',
    ) as reader:
        for e in reader:
            assert np.isclose(e.pointing.array_altitude.to_value(u.deg), 70)
            assert np.isclose(e.pointing.array_azimuth.to_value(u.deg), 0)
            assert np.isnan(e.pointing.array_ra)
            assert np.isnan(e.pointing.array_dec)

            # normal run, alle telescopes point to the array direction
            for pointing in e.pointing.tel.values():
                assert u.isclose(e.pointing.array_azimuth, pointing.azimuth)
                assert u.isclose(e.pointing.array_altitude, pointing.altitude)


def test_allowed_telescopes():
    # test that the allowed_tels mask works:
    allowed_tels = {3, 4}
    with SimTelEventSource(
        input_url=gamma_test_large_path, allowed_tels=allowed_tels, max_events=5,
        focal_length_choice='nominal',
    ) as reader:
        assert not allowed_tels.symmetric_difference(reader.subarray.tel_ids)
        for event in reader:
            assert set(event.r0.tel).issubset(allowed_tels)
            assert set(event.r1.tel).issubset(allowed_tels)
            assert set(event.dl0.tel).issubset(allowed_tels)
            assert set(event.trigger.tels_with_trigger).issubset(allowed_tels)
            assert set(event.pointing.tel).issubset(allowed_tels)


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
    with SimTelEventSource(
        input_url=calib_events_path, skip_calibration_events=False,
        focal_length_choice='nominal',
    ) as reader:

        for event, expected_type in zip_longest(reader, expected_types):
            assert event.trigger.event_type is expected_type


def test_trigger_times():

    source = SimTelEventSource(
        input_url=calib_events_path,
        focal_length_choice='nominal',
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
        focal_length_choice='nominal',
    ) as reader:

        for event in reader:
            for tel in event.simulation.tel.values():
                assert np.count_nonzero(tel.true_image) > 0


def test_instrument():
    """Test if same telescope types share a single instance of CameraGeometry"""
    source = SimTelEventSource(
        input_url=gamma_test_large_path,
        focal_length_choice='nominal',
    )
    subarray = source.subarray
    assert subarray.tel[1].optics.num_mirrors == 1


def test_apply_simtel_r1_calibration_1_channel():
    n_channels = 1
    n_pixels = 2048
    n_samples = 128

    r0_waveforms = np.zeros((n_channels, n_pixels, n_samples))
    pedestal = np.full((n_channels, n_pixels), 20)
    dc_to_pe = np.full((n_channels, n_pixels), 0.5)

    gain_selector = ThresholdGainSelector(threshold=90)
    r1_waveforms, selected_gain_channel = apply_simtel_r1_calibration(
        r0_waveforms, pedestal, dc_to_pe, gain_selector
    )

    assert (selected_gain_channel == 0).all()
    assert r1_waveforms.ndim == 2
    assert r1_waveforms.shape == (n_pixels, n_samples)

    ped = pedestal
    assert r1_waveforms[0, 0] == (r0_waveforms[0, 0, 0] - ped[0, 0]) * dc_to_pe[0, 0]
    assert r1_waveforms[1, 0] == (r0_waveforms[0, 1, 0] - ped[0, 1]) * dc_to_pe[0, 1]


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
    r1_waveforms, selected_gain_channel = apply_simtel_r1_calibration(
        r0_waveforms, pedestal, dc_to_pe, gain_selector
    )

    assert selected_gain_channel[0] == 1
    assert (selected_gain_channel[np.arange(1, 2048)] == 0).all()
    assert r1_waveforms.ndim == 2
    assert r1_waveforms.shape == (n_pixels, n_samples)

    ped = pedestal
    assert r1_waveforms[0, 0] == (r0_waveforms[1, 0, 0] - ped[1, 0]) * dc_to_pe[1, 0]
    assert r1_waveforms[1, 0] == (r0_waveforms[0, 1, 0] - ped[0, 1]) * dc_to_pe[0, 1]


def test_focal_length_choice():
    # this file does not contain the effective focal length
    with pytest.raises(RuntimeError):
        SimTelEventSource(gamma_test_large_path)

    with pytest.raises(RuntimeError):
        SimTelEventSource(gamma_test_large_path, focal_length_choice='effective')

    s = SimTelEventSource(gamma_test_large_path, focal_length_choice='nominal')
    assert s.subarray.tel[1].optics.equivalent_focal_length == 28 * u.m

    # this file does
    s = SimTelEventSource(prod5b_path, focal_length_choice='effective')
    assert u.isclose(s.subarray.tel[1].optics.equivalent_focal_length, 29.3 * u.m, atol=0.05 * u.m)
    # check guessing of the name is not affected by focal length choice
    assert str(s.subarray.tel[1]) == 'LST_LST_LSTCam'

    s = SimTelEventSource(prod5b_path, focal_length_choice='nominal')
    assert u.isclose(s.subarray.tel[1].optics.equivalent_focal_length, 28.0 * u.m, atol=0.05 * u.m)
    # check guessing of the name is not affected by focal length choice
    assert str(s.subarray.tel[1]) == 'LST_LST_LSTCam'



def test_only_config():
    config = Config()
    config.SimTelEventSource.input_url = prod5b_path

    s = SimTelEventSource(config=config)
    assert s.input_url == Path(prod5b_path).absolute()


def test_calibscale_and_calibshift(prod5_gamma_simtel_path):

    telid = 25

    with SimTelEventSource(input_url=prod5_gamma_simtel_path, max_events=1) as source:

        for event in source:
            pass

    calib_scale = 2.0

    with SimTelEventSource(
        input_url=prod5_gamma_simtel_path, max_events=1, calib_scale=calib_scale
    ) as source:

        for event_scaled in source:
            pass

    np.testing.assert_allclose(
        event.r1.tel[telid].waveform[0],
        event_scaled.r1.tel[telid].waveform[0] / calib_scale,
        rtol=0.1,
    )

    calib_shift = 2.0  # p.e.

    with SimTelEventSource(
        input_url=prod5_gamma_simtel_path, max_events=1, calib_shift=calib_shift
    ) as source:

        for event_shifted in source:
            pass

    np.testing.assert_allclose(
        event.r1.tel[telid].waveform[0],
        event_shifted.r1.tel[telid].waveform[0] - calib_shift,
        rtol=0.1,
    )


def test_true_image_sum():
    # this file does not contain true pe info
    with SimTelEventSource(
        gamma_test_large_path,
        focal_length_choice='nominal',
    ) as s:
        e = next(iter(s))
        assert np.all(np.isnan(sim.true_image_sum) for sim in e.simulation.tel.values())

    with SimTelEventSource(
        calib_events_path,
        focal_length_choice='nominal',
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
        calib_events_path, allowed_tels={2, 3},
        focal_length_choice='nominal',
    ) as s:
        e = next(iter(s))
        assert e.simulation.tel[2].true_image_sum == true_image_sums[2]
        assert e.simulation.tel[3].true_image_sum == true_image_sums[3]
