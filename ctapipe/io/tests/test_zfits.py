from pkg_resources import resource_filename
import os

example_file_path = resource_filename(
    'protozfitsreader',
    os.path.join(
        'tests',
        'resources',
        'example_10evts.fits.fz'
    )
)

FIRST_EVENT_NUMBER_IN_FILE = 97750287
ADC_SAMPLES_SHAPE = (1296, 50)


def test_loop_over_events():
    from ctapipe.io.zfits import ZFitsEventSource

    N_EVENTS = 5
    inputfile_reader = ZFitsEventSource(
        input_url=example_file_path,
        max_events=N_EVENTS
    )

    for i, event in enumerate(inputfile_reader):
        assert event.r0.tels_with_data == [1]
        for telid in event.r0.tels_with_data:
            assert (
                event.r0.tel[telid].camera_event_number ==
                FIRST_EVENT_NUMBER_IN_FILE + i
            )

            assert event.r0.tel[telid].adc_samples.shape == ADC_SAMPLES_SHAPE

    # make sure max_events works
    assert i == N_EVENTS - 1
