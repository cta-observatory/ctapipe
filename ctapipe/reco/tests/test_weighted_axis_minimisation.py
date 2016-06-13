from __future__ import print_function, division
from numpy.testing import assert_allclose
from iminuit import Minuit
import numpy as np
from ctapipe.reco.weighted_axis_minimisation import rotate_translate
import astropy.units as u

def f(x, y, z):
    return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

def test_minimiser():
    """
    Simple test of iminuit minimiser to check it is working correctly and
    finding the correct values for a simple function

    """
    m = Minuit(f,x=0,error_x=1,y=1,error_y=1,z=1,error_z=1,errordef=1)
    m.migrad()
    assert_allclose(m.values['x'],2)
    assert_allclose(m.values['y'],3)
    assert_allclose(m.values['z'],4)

def test_rotation():
    """
    Simple test of iminuit minimiser to check it is working correctly and
    finding the correct values for a simple function

    """

    x = np.ones(2048)
    y = np.zeros(2048)
    x,y = rotate_translate(x,y,0,0,90*u.deg)
    assert_allclose(np.sum(x),0,atol=1e-10)
    assert_allclose(np.sum(y),float(y.shape[0]))

test_minimiser()
test_rotation()