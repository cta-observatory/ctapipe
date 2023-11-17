import astropy.units as u
import pytest

from ctapipe.instrument import SubarrayDescription
from ctapipe.utils import get_dataset_path


@pytest.fixture(scope="module")
def subarray(prod5_lst, reference_location):
    tels = [prod5_lst] * 4

    positions = {
        1: [0, 0, 0] * u.m,
        2: [50, 0, 0] * u.m,
        3: [0, 50, 0] * u.m,
        4: [50, 50, 0] * u.m,
    }
    descriptions = {i: t for i, t in enumerate(tels, start=1)}

    return SubarrayDescription(
        "test", positions, descriptions, reference_location=reference_location
    )


def test_toyeventsource(subarray):
    from ctapipe.io.toymodel import ToyEventSource

    s = ToyEventSource(subarray=subarray, max_events=10)

    for i, e in enumerate(s):
        assert e.index.event_id == i
        for tel_id, dl1 in e.dl1.tel.items():
            assert dl1.image.size == subarray.tel[tel_id].camera.geometry.n_pixels
    assert (i + 1) == s.max_events


def test_is_compatible():
    from ctapipe.io.toymodel import ToyEventSource

    assert not ToyEventSource.is_compatible("test.fits.gz")
    assert not ToyEventSource.is_compatible(
        get_dataset_path("gamma_test_large.simtel.gz")
    )
