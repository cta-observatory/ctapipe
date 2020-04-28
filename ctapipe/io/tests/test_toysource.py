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

    for i, e in enumerate(s):
        assert e.index.event_id == i
        for tel_id, dl1 in e.dl1.tel.items():
            assert dl1.image.size == baseline_array.tel[tel_id].camera.geometry.n_pixels
    assert (i + 1) == s.max_events
