import numpy as np

f = lambda x: 4./(1. + x*x)

def compPi(niter=1000):
    h  = 1./niter
    pi = 0.
    for i in range(niter):
        x   = h*(i - 0.5)
        pi += f(x)
    error = abs(np.arccos(-1.) - pi*h)/np.arccos(-1.)
    return pi*h, error