import numpy as np
import pytest
import copy
from ctapipe.utils import get_dataset_path
from ctapipe.io.simteleventsource import SimTelEventSource
from ctapipe.io.simteleventsource import HESSIOEventSource

dataset_path = get_dataset_path("gamma_test_large.simtel.gz")

def test_compare_5_event_hessio_and_simtel():
    kwargs = dict(config=None, tool=None, input_url=dataset_path)

    with SimTelEventSource(**kwargs) as simtel_source:
        iter_simtel_event = iter(simtel_source)

        with HESSIOEventSource(**kwargs) as hessio_source:
            iter_hessio_event = iter(hessio_source)

            for test_event in range(5):
                simtel_event = next(iter_simtel_event)
                hessio_event = next(iter_hessio_event)

                s = simtel_event # .as_dict(recursive=True, flatten=True)
                h = hessio_event # .as_dict(recursive=True, flatten=True)

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

                    assert (h.mc.tel[tel_id].dc_to_pe == s.mc.tel[tel_id].dc_to_pe).all()
                    assert (h.mc.tel[tel_id].pedestal == s.mc.tel[tel_id].pedestal).all()
                    assert h.r0.tel[tel_id].waveform.shape == s.r0.tel[tel_id].waveform.shape
                    assert np.allclose(h.r0.tel[tel_id].waveform, s.r0.tel[tel_id].waveform)
                    assert (h.r0.tel[tel_id].num_samples == s.r0.tel[tel_id].num_samples)
                    assert (h.r0.tel[tel_id].image == s.r0.tel[tel_id].image).all()

                    # assert h.r0.tel[tel_id].num_trig_pix == s.r0.tel[tel_id].num_trig_pix
                    # assert h.r0.tel[tel_id].trig_pix_id == s.r0.tel[tel_id].trig_pix_id
                    assert (h.mc.tel[tel_id].reference_pulse_shape == s.mc.tel[tel_id].reference_pulse_shape).all()

                    # assert (h.mc.tel[tel_id].photo_electron_image == s.mc.tel[tel_id].photo_electron_image).all()
                    assert h.mc.tel[tel_id].meta == s.mc.tel[tel_id].meta
                    assert h.mc.tel[tel_id].time_slice == s.mc.tel[tel_id].time_slice
                    # assert h.mc.tel[tel_id].azimuth_raw == s.mc.tel[tel_id].azimuth_raw
                    # assert h.mc.tel[tel_id].altitude_raw == s.mc.tel[tel_id].altitude_raw



@pytest.mark.xfail
def test_hessio_file_reader():
    kwargs = dict(config=None, tool=None, input_url=dataset_path)

    with SimTelEventSource(**kwargs) as reader:
        assert reader.is_compatible(dataset_path)
        assert not reader.is_stream

        for event in reader:
            if event.count == 0:
                assert event.r0.tels_with_data == {38, 47}
            elif event.count == 1:
                assert event.r0.tels_with_data == {11, 21, 24, 26, 61, 63, 118,
                                                   119}
            else:
                break
        for event in reader:
            # Check generator has restarted from beginning
            assert event.count == 0
            break

    # test that max_events works:
    max_events = 5
    with SimTelEventSource(**kwargs, max_events=max_events) as reader:
        count = 0
        for _ in reader:
            count += 1
        assert count == max_events

    # test that the allowed_tels mask works:
    with SimTelEventSource(**kwargs, allowed_tels={3, 4}) as reader:
        for event in reader:
            assert event.r0.tels_with_data.issubset(reader.allowed_tels)

"""
@pytest.mark.xfail
def test_that_event_is_not_modified_after_loop():

    dataset = get_dataset_path("gamma_test.simtel.gz")
    with SimTelEventSource(input_url=dataset, max_events=2) as source:
        for event in source:
            last_event = copy.deepcopy(event)

        # now `event` should be identical with the deepcopy of itself from
        # inside the loop.
        # Unfortunately this does not work:
        #      assert last_event == event
        # So for the moment we just compare event ids
        assert event.r0.event_id == last_event.r0.event_id
"""
