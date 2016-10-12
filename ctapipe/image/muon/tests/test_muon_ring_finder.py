from ctapipe.image.muon import muon_ring_finder
import numpy as np

def test_ChaudhuriKunduRingFitter():

    fitter = muon_ring_finder.ChaudhuriKunduRingFitter()

    x = np.linspace(-100,100,200)
    y = np.linspace(-100,100,200)

    XX,YY = np.meshgrid(x,y)
    ZZ = np.zeros_like(XX)
    r =  np.sqrt((XX-50)**2+(YY-20)**2)
    ZZ[(r>10) & (r<20)]=1

    c_x,c_y,r = fitter.fit(XX,YY,ZZ)
    
    print(c_x,c_y,r)

    assert(abs(c_x-50)<0.05 and abs(c_y-20)<0.05 and abs(r-15)<1)

