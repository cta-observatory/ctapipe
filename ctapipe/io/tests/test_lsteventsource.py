from pkg_resources import resource_filename
import os

import pytest
pytest.importorskip("protozfits", minversion="1.0.2")

example_file_path = resource_filename(
    'protozfits',
    os.path.join(
        'tests',
        'resources',
        'example_LST_R1_10_evts.fits.fz'
    )
)

FIRST_EVENT_NUMBER_IN_FILE = 1
ADC_SAMPLES_SHAPE = (2, 14, 40)


def test_loop_over_events():
    from ctapipe.io.lsteventsource import LSTEventSource

    N_EVENTS = 10
    inputfile_reader = LSTEventSource(
        input_url=example_file_path,
        max_events=N_EVENTS
    )

    for i, event in enumerate(inputfile_reader):
        assert event.r0.tels_with_data == [0]
        for telid in event.r0.tels_with_data:
            assert event.r0.event_id == FIRST_EVENT_NUMBER_IN_FILE + i
            assert event.r0.tel[telid].waveform.shape == ADC_SAMPLES_SHAPE

    # make sure max_events works
    assert i == N_EVENTS - 1


def test_is_compatible():
    from ctapipe.io.lsteventsource import LSTEventSource

    assert LSTEventSource.is_compatible(example_file_path)


def test_factory_for_lst_file():
    from ctapipe.io.eventsourcefactory import EventSourceFactory
    from ctapipe.io.lsteventsource import LSTEventSource

    reader = EventSourceFactory.produce(input_url=example_file_path)
    assert isinstance(reader, LSTEventSource)
    assert reader.input_url == example_file_path
