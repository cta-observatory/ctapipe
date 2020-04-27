import pytest
from ctapipe.instrument import SubarrayDescription, TelescopeDescription
import numpy as np
import astropy.units as u


@pytest.fixture(scope='module')
def baseline_array():

    lst = TelescopeDescription.from_name('LST', 'LSTCam')
    tels = [lst] * 4

    positions = {i: np.zeros(3) * u.m for i, t in enumerate(tels, start=1)}
    descriptions = {i: t for i, t in enumerate(tels, start=1)}

    return SubarrayDescription('test', positions, descriptions)


def test_toyeventsource(baseline_array):
    from ctapipe.io.toymodel import ToyEventSource

    s = ToyEventSource(subarray=baseline_array, max_events=10)

    import matplotlib.pyplot as plt
    fig = plt.figure()

    for i, e in enumerate(s, start=1):
        for tel_id, dl1 in e.dl1.tel.items():
            cam = baseline_array.tel[tel_id].camera.geometry

            from ctapipe.visualization import CameraDisplay

            fig.clf()
            ax = fig.add_subplot(1, 1, 1)
            d = CameraDisplay(cam, ax=ax, image=dl1.image, cmap='inferno')
            d.add_colorbar(ax=ax)
            d.axes.figure.show()
            plt.pause(0.1)

    i == s.max_events
