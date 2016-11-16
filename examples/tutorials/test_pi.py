import math
from calcpy import compPi


def test_result(niter=1000):
    prec = 1./niter
    pi, error = compPi(niter)
    assert abs(math.pi - pi)/math.pi < prec
    
def test_prec_result(niter=1000):
    prec = 1./niter
    pi, error = compPi(niter)
    assert error < prec
    
def test_result_logical_error(niter=1000):
    prec = 1./niter
    pi, error = compPi(1j)
    assert abs(math.pi - pi)/math.pi < prec
    
def test_prec_result_failure(niter=1000):
    prec = 1./niter
    pi, error = compPi(niter)
    assert error > prec