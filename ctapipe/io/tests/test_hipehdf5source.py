from pkg_resources import resource_filename
import os
import numpy as np

import pytest
import tables

from ctapipe.utils import get_dataset_path

example_file_path = get_dataset_path('gamma_test_large_hipecta.h5')
FIRST_EVENT_NUMBER_IN_TEL = [153614, 31012, 31012, 31012, 23703, 31012, 46705, 31007,
135104, 31012, 31007, 31012, 31012, 23703, 46705, 135104, 229110, 23703, 277618, 46705,
31007, 135104, 394306, 90914, 135104, 135104, 135104, 23703, 46705, 31007, 1864717, 394306,
869914, 277617, 31012, 135104, 31007, 397303, 394306, 135104, 31010, 277618, 1675711, 1625719,
31012, 397308, 1221503, 1883517, 1463300, 1044706, 135107, 135104, 31010, 904414, 1221503, 31007,
1044706, 1096805, 1439710, 135104, 135101, 31010, 1625713, 1463300, 31012, 904404, 135101, 1439706,
1675711, 31019, 880611, 1625713, 904414, 135107, 31012, 1044706, 135107, 1147119, 135101, 880603,
394309, 904404, 1221503, 31012, 135101, 880603, 394309, 1439710, 31019, 880611]


def test_compare_with_simtel():
	from ctapipe.io import event_source
	hipefile = get_dataset_path('gamma_test_large_hipecta.h5')
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
	#assert(len(hipe_event.inst.subarray.tels) == len(sim_event.inst.subarray.tels))

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
	inputfile_reader = HiPeHDF5EventSource(input_url=example_file_path)
	i = 0
	for tel in inputfile_reader.run.walk_nodes('/Tel', 'Group'):
		try:
			tabEventId = tel.eventId.read()
			tabEventId = tabEventId["eventId"]
			assert (tabEventId[0] == FIRST_EVENT_NUMBER_IN_TEL[i])
			i += 1
		except tables.exceptions.NoSuchNodeError as e:
			pass


def test_is_compatible():
	from ctapipe.io.hipehdf5eventsource import HiPeHDF5EventSource
	assert HiPeHDF5EventSource.is_compatible(example_file_path)



