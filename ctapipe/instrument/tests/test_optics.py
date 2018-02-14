from ctapipe.instrument.optics import OpticsDescription
from astropy import units as u
import pytest


def test_guess_optics():
    od = OpticsDescription.guess(28.0 * u.m)
    od.info()

    assert od.tel_type == 'LST'
    assert od.tel_subtype == ''
    assert od.mirror_type == 'DC'

    with pytest.raises(KeyError):
        OpticsDescription.guess(0 * u.m)  # unknown tel


def test_construct_optics():
    with pytest.raises(ValueError):
        OpticsDescription(mirror_type="DC",
                          tel_type="bad",  # bad value
                          tel_subtype="1M",
                          equivalent_focal_length=10 * u.m)

    with pytest.raises(ValueError):
        OpticsDescription(mirror_type="bad",  # bad value
                          tel_type="MST",
                          tel_subtype="1M",
                          equivalent_focal_length=10 * u.m)

    with pytest.raises(u.UnitsError):
        OpticsDescription.guess(28.0 * u.kg)  # bad unit

    with pytest.raises(TypeError):
        OpticsDescription.guess(28.0)  # not a unit quantity
