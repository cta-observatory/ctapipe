from ctapipe.image.muon import muon_ring_finder
import numpy as np
import astropy.units as u

def test_ChaudhuriKunduRingFitter():

    fitter = muon_ring_finder.ChaudhuriKunduRingFitter()

    x = np.linspace(-100,100,200)
    y = np.linspace(-100,100,200)

    XX,YY = np.meshgrid(x,y) * u.deg
    ZZ    = np.zeros_like(XX)

    c_x   = 50*u.deg
    c_y   = 20*u.deg

    r     =  np.sqrt((XX-c_x)**2+(YY-c_y)**2) 

    min_r = 10 *u.deg
    max_r = 20 *u.deg

    ZZ[(r>min_r) & (r<max_r)] = 1 * u.deg
    output = fitter.fit(XX,YY,ZZ)
#    print(output)

    lim_p = 0.05 * u.deg
    lim_r = 1 * u.deg
    rad_a = 0.5*(max_r+min_r)

    assert(abs(output.ring_center_x-c_x)<lim_p and (abs(output.ring_center_y-c_y)<lim_p) and (abs(output.ring_radius-rad_a)<lim_r))

