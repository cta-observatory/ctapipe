from ...io.files import InputFile
from ...utils.datasets import get_datasets_path
from ..camera import CameraPlotter
import numpy as np


def test_eventplotter():
    dataset = get_datasets_path("gamma_test.simtel.gz")
    file = InputFile(dataset, 'hessio')
    source = file.read()
    event = next(source)
    data = event.dl0.tel[38].adc_samples[0]
    plotter = CameraPlotter(event)

    camera = plotter.draw_camera(38, data[:, 0])
    assert camera is not None
    np.testing.assert_array_equal(camera.image, data[:, 0])

    plotter.draw_camera_pixel_ids(38, [0, 1, 2])

    waveform = plotter.draw_waveform(data[0, :])
    assert waveform is not None
    np.testing.assert_array_equal(waveform.get_ydata(), data[0, :])

    line = plotter.draw_waveform_positionline(0)
    assert line is not None
    np.testing.assert_array_equal(line.get_xdata(), [0, 0])
