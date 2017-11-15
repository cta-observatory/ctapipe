from ctapipe.instrument.optics import  OpticsDescription
from astropy import units as u
import pytest

def test_guess_optics():

    od = OpticsDescription.guess(28.0*u.m)
    od.info()

    assert od.tel_type == 'LST'
    assert od.tel_subtype == ''
    assert od.mirror_type == 'DC'

    with pytest.raises(KeyError):
        od2 = OpticsDescription.guess(0*u.m)


