from ctapipe.instrument.optics import  OpticsDescription
from astropy import units as u

def test_guess_optics():

    od = OpticsDescription.guess(28.0*u.m)

    assert od.tel_type == 'LST'
    assert od.tel_subtype == ''
    assert od.mirror_type == 'DC'

    od2 = OpticsDescription.guess(0*u.m)

    assert od2.tel_type == 'unknown'