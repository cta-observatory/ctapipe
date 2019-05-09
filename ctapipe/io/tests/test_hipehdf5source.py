from pkg_resources import resource_filename
import os
import numpy as np

import pytest

from ctapipe.utils import get_dataset_path

example_file_path = get_dataset_path('gamma_test_large.h5')
FIRST_EVENT_NUMBER_IN_TEL = [4907, 4907, 9508, 4907, 10104, 12202, 9508, 409,
                             9508, 4907, 4907, 10104, 803, 4907, 409, 409,
                             9508, 409, 10109, 408, 408, 409, 12202, 409,
                             12202, 12202, 803, 12202, 12202, 409, 409,
                             803, 12202]


def test_compare_with_simtel():
    from ctapipe.io import event_source
    hipefile = get_dataset_path('gamma_test_large.h5')
    simfile = get_dataset_path('gamma_test_large.simtel.gz')
    event_id = 880610
    sim_source = event_source(simfile)
    hipe_source = event_source(hipefile)

    for sim_event in sim_source:
        if sim_event.r0.event_id == event_id:
            break

    for hipe_event in hipe_source:
        if hipe_event.r0.event_id == event_id :
            break

    assert(hipe_event.meta['origin'] == "hipehdf5")
    assert(hipe_event.meta['input_url'] == hipefile)
    assert(hipe_event.meta['max_events'] == None)

    assert(hipe_event.inst.subarray.name == sim_event.inst.subarray.name)
    assert(len(hipe_event.inst.subarray.tels) ==
           len(sim_event.inst.subarray.tels))

    assert (hipe_event.r0.event_id == sim_event.r0.event_id)
    assert (hipe_event.r0.tels_with_data == sim_event.r0.tels_with_data)
    #assert (hipe_event.r0.obs_id == sim_event.r0.obs_id)

    assert (hipe_event.r1.event_id == sim_event.r1.event_id)
    assert (hipe_event.r1.tels_with_data == sim_event.r1.tels_with_data)
    #assert (hipe_event.r1.obs_id == sim_event.r1.obs_id)

    assert (hipe_event.dl0.tels_with_data == sim_event.dl0.tels_with_data)
    assert (hipe_event.dl0.event_id == sim_event.dl0.event_id)
    #assert(hipe_event.dl0.obs_id == sim_event.dl0.obs_id)

    assert(hipe_event.trig.tels_with_trigger.all() ==
           sim_event.trig.tels_with_trigger.all())


    for tel_id in  sim_event.r0.tels_with_data:
        assert(hipe_event.r0.tel[tel_id].waveform.sum() ==
               sim_event.r0.tel[tel_id].waveform.sum())
        assert(hipe_event.r0.tel[tel_id].image.sum() ==
               sim_event.r0.tel[tel_id].image.sum())
        assert(hipe_event.r0.tel[tel_id].num_samples ==
               sim_event.r0.tel[tel_id].num_samples)


    for tel_id in sim_event.mc.tel.keys():
        # Could not compare pixel by pixel because pixel order differ
        for gain in range(len(sim_event.mc.tel[tel_id].dc_to_pe)):
            assert (np.isclose(sim_event.mc.tel[tel_id].dc_to_pe[gain].sum(),
                               hipe_event.mc.tel[tel_id].dc_to_pe[gain].sum()))
            assert (np.isclose(sim_event.mc.tel[tel_id].pedestal[gain].sum(),
                               hipe_event.mc.tel[tel_id].pedestal[gain].sum()))
            assert (np.isclose(sim_event.mc.tel[tel_id].
                               reference_pulse_shape[gain].sum(),
                               hipe_event.mc.tel[tel_id].
                               reference_pulse_shape[gain].sum()))







def test_loop_over_telescopes():
    from ctapipe.io.hipehdf5eventsource import HiPeHDF5EventSource

    n_events = 10
    inputfile_reader = HiPeHDF5EventSource(
        input_url=example_file_path,
        max_events=n_events
    )
    i = 0
    for tel in fileIn.walk_nodes('/Tel', 'Group'):
        tabEventId = tel.eventId.read()
        tabEventId = ["tabEventId"]
        assert (tabEventId[0] == FIRST_EVENT_NUMBER_IN_TEL[i])
        i += 1


def test_is_compatible():
    from ctapipe.io.hipehdf5eventsource import HiPeHDF5EventSource
    assert HiPeHDF5EventSource.is_compatible(example_file_path)



