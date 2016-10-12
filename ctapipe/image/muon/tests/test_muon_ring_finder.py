from ctapipe.image.muon import muon_ring_finder
import numpy as np

def test_ChaudhuriKunduRingFitter():

    fitter = muon_ring_finder.ChaudhuriKunduRingFitter()

    x = np.linspace(-100,100,200)
    y = np.linspace(-100,100,200)
    w = np.random.normal(1,1,200)

    c_x,c_y,r = fitter.fit(x,y,w)
    
    assert(c_x!=0 and c_y!=0 and r!=0)
