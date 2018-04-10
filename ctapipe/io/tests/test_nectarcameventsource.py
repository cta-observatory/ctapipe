from pkg_resources import resource_filename
import os

import pytest
pytest.importorskip("protozfits", minversion="0.44.4")

example_file_path = resource_filename(
    'protozfits',
    os.path.join(
        'tests',
        'resources',
        'example_9evts_NectarCAM.fits.fz'
    )
)

FIRST_EVENT_NUMBER_IN_FILE = 1
ADC_SAMPLES_SHAPE = (2, 84, 60)


def test_loop_over_events():
    from ctapipe.io.nectarcameventsource import NectarCAMEventSource

    N_EVENTS = 3
    inputfile_reader = NectarCAMEventSource(
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
    from ctapipe.io.nectarcameventsource import NectarCAMEventSource

    assert NectarCAMEventSource.is_compatible(example_file_path)


def test_factory_for_nectarcam_file():
    from ctapipe.io.eventsourcefactory import EventSourceFactory
    from ctapipe.io.nectarcameventsource import NectarCAMEventSource

    reader = EventSourceFactory.produce(input_url=example_file_path)
    assert isinstance(reader, NectarCAMEventSource)
    assert reader.input_url == example_file_path
