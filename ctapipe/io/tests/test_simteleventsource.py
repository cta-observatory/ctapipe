import numpy as np
import pytest
import copy
from ctapipe.utils import get_dataset_path
from ctapipe.io.simteleventsource import SimTelEventSource, apply_simtel_r1_calibration
from ctapipe.io.hessioeventsource import HESSIOEventSource
from ctapipe.calib.camera.gainselection import ThresholdGainSelector
from itertools import zip_longest
from copy import deepcopy

gamma_test_large_path = get_dataset_path("gamma_test_large.simtel.gz")
gamma_test_path = get_dataset_path("gamma_test.simtel.gz")
calib_events_path = get_dataset_path('calib_events.simtel.gz')


def compare_sources(input_url):
    pytest.importorskip('pyhessio')

    with SimTelEventSource(input_url=input_url) as simtel_source, \
            HESSIOEventSource(input_url=input_url) as hessio_source:

        for s, h in zip_longest(simtel_source, hessio_source):

            assert s is not None
            assert h is not None

            assert h.count == s.count
            assert h.r0.obs_id == s.r0.obs_id
            assert h.r0.event_id == s.r0.event_id
            assert h.r0.tels_with_data == s.r0.tels_with_data

            assert (h.trig.tels_with_trigger == s.trig.tels_with_trigger).all()
            assert h.trig.gps_time == s.trig.gps_time

            assert h.mc.energy == s.mc.energy
            assert h.mc.alt == s.mc.alt
            assert h.mc.az == s.mc.az
            assert h.mc.core_x == s.mc.core_x
            assert h.mc.core_y == s.mc.core_y

            assert h.mc.h_first_int == s.mc.h_first_int
            assert h.mc.x_max == s.mc.x_max
            assert h.mc.shower_primary_id == s.mc.shower_primary_id
            assert (h.mcheader.run_array_direction == s.mcheader.run_array_direction).all()

            tels_with_data = s.r0.tels_with_data
            for tel_id in tels_with_data:

                assert h.mc.tel[tel_id].reference_pulse_shape.dtype == s.mc.tel[tel_id].reference_pulse_shape.dtype
                assert type(h.mc.tel[tel_id].meta['refstep']) is type(s.mc.tel[tel_id].meta['refstep'])
                assert type(h.mc.tel[tel_id].time_slice) is type(s.mc.tel[tel_id].time_slice)

                assert (h.mc.tel[tel_id].dc_to_pe == s.mc.tel[tel_id].dc_to_pe).all()
                assert (h.mc.tel[tel_id].pedestal == s.mc.tel[tel_id].pedestal).all()
                assert h.r0.tel[tel_id].waveform.shape == s.r0.tel[tel_id].waveform.shape
                assert h.r1.tel[tel_id].waveform.shape == s.r1.tel[tel_id].waveform.shape
                assert np.allclose(h.r0.tel[tel_id].waveform, s.r0.tel[tel_id].waveform)
                assert np.allclose(h.r1.tel[tel_id].waveform, s.r1.tel[tel_id].waveform)

                assert h.r0.tel[tel_id].num_trig_pix == s.r0.tel[tel_id].num_trig_pix
                assert (h.r0.tel[tel_id].trig_pix_id == s.r0.tel[tel_id].trig_pix_id).all()
                assert (h.mc.tel[tel_id].reference_pulse_shape == s.mc.tel[tel_id].reference_pulse_shape).all()

                assert (h.mc.tel[tel_id].photo_electron_image == s.mc.tel[tel_id].photo_electron_image).all()
                assert h.mc.tel[tel_id].meta == s.mc.tel[tel_id].meta
                assert h.mc.tel[tel_id].time_slice == s.mc.tel[tel_id].time_slice
                assert h.mc.tel[tel_id].azimuth_raw == s.mc.tel[tel_id].azimuth_raw
                assert h.mc.tel[tel_id].altitude_raw == s.mc.tel[tel_id].altitude_raw
                assert h.pointing[tel_id].altitude == s.pointing[tel_id].altitude
                assert h.pointing[tel_id].azimuth == s.pointing[tel_id].azimuth

                assert (h.inst.subarray.tel[tel_id].camera.sampling_rate ==
                        s.inst.subarray.tel[tel_id].camera.sampling_rate)


def test_compare_event_hessio_and_simtel():
    compare_sources(gamma_test_large_path)


def test_simtel_event_source_on_gamma_test_one_event():
    with SimTelEventSource(input_url=gamma_test_large_path, back_seekable=True) as reader:
        assert reader.is_compatible(gamma_test_large_path)
        assert not reader.is_stream

        for event in reader:
            if event.count > 1:
                break

        for event in reader:
            # Check generator has restarted from beginning
            assert event.count == 0
            break

    # test that max_events works:
    max_events = 5
    with SimTelEventSource(input_url=gamma_test_large_path, max_events=max_events) as reader:
        count = 0
        for _ in reader:
            count += 1
        assert count == max_events

    # test that the allowed_tels mask works:
    with SimTelEventSource(
        input_url=gamma_test_large_path,
        allowed_tels={3, 4}
    ) as reader:
        for event in reader:
            assert event.r0.tels_with_data.issubset(reader.allowed_tels)


def test_that_event_is_not_modified_after_loop():

    dataset = gamma_test_large_path
    with SimTelEventSource(input_url=dataset, max_events=2) as source:
        for event in source:
            last_event = copy.deepcopy(event)

        # now `event` should be identical with the deepcopy of itself from
        # inside the loop.
        # Unfortunately this does not work:
        #      assert last_event == event
        # So for the moment we just compare event ids
        assert event.r0.event_id == last_event.r0.event_id


def test_additional_meta_data_from_mc_header():
    with SimTelEventSource(input_url=gamma_test_large_path) as reader:
        data = next(iter(reader))

    # for expectation values
    from astropy import units as u
    from astropy.coordinates import Angle

    assert data.mcheader.corsika_version == 6990
    assert data.mcheader.spectral_index == -2.0
    assert data.mcheader.shower_reuse == 20
    assert data.mcheader.core_pos_mode == 1
    assert data.mcheader.diffuse == 1
    assert data.mcheader.atmosphere == 26

    # value read by hand from input card
    name_expectation = {
        'energy_range_min': u.Quantity(3.0e-03, u.TeV),
        'energy_range_max': u.Quantity(3.3e+02, u.TeV),
        'prod_site_B_total': u.Quantity(23.11772346496582, u.uT),
        'prod_site_B_declination': Angle(0.0 * u.rad),
        'prod_site_B_inclination': Angle(-0.39641156792640686 * u.rad),
        'prod_site_alt': 2150.0 * u.m,
        'max_scatter_range': 3000.0 * u.m,
        'min_az': 0.0 * u.rad,
        'min_alt': 1.2217305 * u.rad,
        'max_viewcone_radius': 10.0 * u.deg,
        'corsika_wlen_min': 240 * u.nm,

    }

    for name, expectation in name_expectation.items():
        value = getattr(data.mcheader, name)

        assert value.unit == expectation.unit
        assert np.isclose(
            value.to_value(expectation.unit),
            expectation.to_value(expectation.unit)
        )


def test_hessio_file_reader():
    dataset = gamma_test_path

    with SimTelEventSource(input_url=dataset) as reader:
        assert reader.is_compatible(dataset)
        assert reader.is_stream  # using gzip subprocess makes it a stream

        for event in reader:
            if event.count == 0:
                assert event.r0.tels_with_data == {38, 47}
            elif event.count == 1:
                assert event.r0.tels_with_data == {11, 21, 24, 26, 61, 63, 118,
                                                   119}
            else:
                break

    # test that max_events works:
    max_events = 5
    with SimTelEventSource(input_url=dataset, max_events=max_events) as reader:
        count = 0
        for _ in reader:
            count += 1
        assert count == max_events

    # test that the allowed_tels mask works:
    with SimTelEventSource(input_url=dataset, allowed_tels={3, 4}) as reader:
        for event in reader:
            assert event.r0.tels_with_data.issubset(reader.allowed_tels)


def test_calibration_events():
    with SimTelEventSource(
            input_url=calib_events_path,
            skip_calibration_events=False,
    ) as reader:
        for e in reader:
            pass


def test_camera_caching():
    '''Test if same telescope types share a single instance of CameraGeometry'''
    source = SimTelEventSource(input_url=gamma_test_large_path)
    event = next(iter(source))
    subarray = event.inst.subarray
    assert subarray.tel[1].camera is subarray.tel[2].camera


def test_instrument():
    '''Test if same telescope types share a single instance of CameraGeometry'''
    source = SimTelEventSource(input_url=gamma_test_large_path)
    event = next(iter(source))
    subarray = event.inst.subarray
    assert subarray.tel[1].optics.num_mirrors == 1


def test_subarray_property():
    source = SimTelEventSource(input_url=gamma_test_large_path)
    subarray = deepcopy(source.subarray)
    event = next(iter(source))
    subarray_event = event.inst.subarray
    assert subarray.tel.keys() == subarray_event.tel.keys()
    assert (subarray.tel[1].camera.pix_x ==
            subarray_event.tel[1].camera.pix_x).all()


def test_apply_simtel_r1_calibration_1_channel():
    n_channels = 1
    n_pixels = 2048
    n_samples = 128

    r0_waveforms = np.zeros((n_channels, n_pixels, n_samples))
    pedestal = np.full((n_channels, n_pixels), 20 * n_samples)
    dc_to_pe = np.full((n_channels, n_pixels), 0.5)

    gain_selector = ThresholdGainSelector(threshold=90)
    r1_waveforms, selected_gain_channel = apply_simtel_r1_calibration(
        r0_waveforms, pedestal, dc_to_pe, gain_selector
    )

    assert (selected_gain_channel == 0).all()
    assert r1_waveforms.ndim == 2
    assert r1_waveforms.shape == (n_pixels, n_samples)

    ped = pedestal / n_samples
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
    pedestal[0] = 90 * n_samples
    pedestal[1] = 0.9 * n_samples

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

    ped = pedestal / n_samples
    assert r1_waveforms[0, 0] == (r0_waveforms[1, 0, 0] - ped[1, 0]) * dc_to_pe[1, 0]
    assert r1_waveforms[1, 0] == (r0_waveforms[0, 1, 0] - ped[0, 1]) * dc_to_pe[0, 1]
