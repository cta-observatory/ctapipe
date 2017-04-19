from __future__ import print_function, division
from numpy.testing import assert_allclose
from iminuit import Minuit
import numpy as np
from ctapipe.reco.weighted_axis_minimisation import WeightedAxisMinimisation

import astropy.units as u


def test_minimiser():
    """
    Simple test of iminuit minimiser to check it is working correctly and
    finding the correct values for a simple function

    """
    def f(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    m = Minuit(f,x=0,error_x=1,y=1,error_y=1,z=1,error_z=1,errordef=1)
    m.migrad()
    assert_allclose(m.values['x'],2)
    assert_allclose(m.values['y'],3)
    assert_allclose(m.values['z'],4)


def test_rotation():
    """
    Test of image rotation function. Create 2 perpendicular images and
    rotate them by 90 deg

    """

    x = np.ones(2048)
    y = np.zeros(2048)
    w = weighted_axis_minimisation()

    x,y = w.rotate_translate(x,y,0,0,90*u.deg)
    assert_allclose(np.sum(x),0,atol=1e-10)
    assert_allclose(np.sum(y),float(y.shape[0]))


def test_minimisation():
    """
    Test of full minimisation method, creates 2 perpendicular images
    and checks the reconstructed source and ground positions are [0,0]

    """
    reco = WeightedAxisMinimisation()

    x_pix_1 = np.arange(4)
    y_pix_1 = np.zeros(4)

    y_pix_2 = np.arange(4)
    x_pix_2 = np.zeros(4)

    x_pix = [x_pix_1,x_pix_2]
    y_pix = [y_pix_1,y_pix_2]

    tel_x = [100,0]
    tel_y = [0,100]

    shower = reco.reconstruct_event([0,0],tel_x,tel_y,x_pix,y_pix,[1,1])

    assert_allclose(shower['x_src'],0,atol=1e-10)
    assert_allclose(shower['y_src'],0,atol=1e-10)
    assert_allclose(shower['x_grd'],0,atol=1e-10)
    assert_allclose(shower['y_grd'],0,atol=1e-10)


test_minimiser()
test_rotation()
test_minimisation()