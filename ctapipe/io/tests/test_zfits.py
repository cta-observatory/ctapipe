from ctapipe.utils.datasets import get_dataset
import numpy as np
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


def test_loop_over_events():
    from ctapipe.io.zfits import zfits_event_source

    inputfile_reader = zfits_event_source(url=example_file_path, max_events=5)

    for i, event in enumerate(inputfile_reader):
        tels = event.r0.tels_with_data
        assert tels == [3]
        for telid in event.r0.tels_with_data:
            evt_num = event.r0.tel[telid].camera_event_number
            assert i == evt_num
            adcs = np.array(list(event.r0.tel[telid].adc_samples.values()))
            assert adcs.shape == (1296, 20)
