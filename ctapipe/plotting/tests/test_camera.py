from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import get_dataset
from ctapipe.plotting.camera import CameraPlotter
import numpy as np


def test_eventplotter():
    dataset = get_dataset("gamma_test.simtel.gz")
    source = hessio_event_source(dataset, max_events=1)
    event = next(source)
    del source

    telid = list(event.r0.tels_with_data)[0]

    data = event.r0.tel[telid].waveform[0]
    plotter = CameraPlotter(event)

    camera = plotter.draw_camera(telid, data[:, 0])
    assert camera is not None
    np.testing.assert_array_equal(camera.image, data[:, 0])

    plotter.draw_camera_pixel_ids(telid, [0, 1, 2])

    waveform = plotter.draw_waveform(data[0, :])
    assert waveform is not None
    np.testing.assert_array_equal(waveform.get_ydata(), data[0, :])

    line = plotter.draw_waveform_positionline(0)
    assert line is not None
    np.testing.assert_array_equal(line.get_xdata(), [0, 0])

