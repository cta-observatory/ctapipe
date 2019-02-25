from pkg_resources import resource_filename
import os

import pytest
from ctapipe.utils import get_dataset_path
pytest.importorskip("protozfits", minversion="1.4.2")

FIRST_EVENT_NUMBER_IN_FILE = 1

# example_file_path="NectarCam.Run0890.10events.fits.fz"
example_file_path = get_dataset_path("NectarCAM.Run0890.10events.fits.fz")

def test_loop_over_events():
    from ctapipe.io.nectarcameventsource import NectarCAMEventSource

    n_events = 10
    inputfile_reader = NectarCAMEventSource(
        input_url=example_file_path,
        max_events=n_events
    )

    for i, event in enumerate(inputfile_reader):
        assert event.r0.tels_with_data == [0]
        for telid in event.r0.tels_with_data:
            assert event.r0.event_id == FIRST_EVENT_NUMBER_IN_FILE + i
            n_gain = 2
            n_camera_pixels = event.inst.subarray.tels[telid].camera.n_pixels

            num_samples = event.nectarcam.tel[telid].svc.num_samples
            waveform_shape = (n_gain, n_camera_pixels, num_samples)
            assert event.r0.tel[telid].waveform.shape == waveform_shape

    # make sure max_events works
    assert i == n_events - 1


def test_is_compatible():
    from ctapipe.io.nectarcameventsource import NectarCAMEventSource

    assert NectarCAMEventSource.is_compatible(example_file_path)


def test_factory_for_nectarcam_file():
    from ctapipe.io import event_source
    from ctapipe.io.nectarcameventsource import NectarCAMEventSource

    reader = event_source(example_file_path)
    assert isinstance(reader, NectarCAMEventSource)
    assert reader.input_url == example_file_path
