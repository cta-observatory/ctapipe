import numpy as np
import pkg_resources
import os
from glob import glob

from protozfits import File

example_file_path = pkg_resources.resource_filename(
    'protozfits',
    os.path.join(
        'tests',
        'resources',
        'example_LST_R1_10_evts.fits.fz'
    )
)


def test_can_iterate_over_events_and_run_header():

    with File(example_file_path) as f:

        camera_config = next(f.CameraConfig)
        assert (camera_config.expected_pixels_id == np.arange(14)).all()

        for i, event in enumerate(f.Events):
            assert event.event_id == i + 1
            assert event.waveform.shape == (1120,)
            assert event.pixel_status.shape == (14,)
            assert event.lstcam.first_capacitor_id.shape == (16,)
            assert event.lstcam.counters.shape == (44,)


glob_path = pkg_resources.resource_filename(
    'protozfits',
    os.path.join(
        'tests',
        'resources',
        '*.fits.fz'
    )
)

all_test_resources = sorted(glob(glob_path))


def test_can_open_and_get_an_event_from_all_test_resources():
    print()
    for path in all_test_resources:
        with File(path) as f:
            event = next(f.Events)
        print(path, len(str(event)))
