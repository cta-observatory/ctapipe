from ..shower_max import ShowerMaxEstimator
from astropy import units as u
from ctapipe.utils.datasets import get_path


def test_showermaxestimator():
    estim = ShowerMaxEstimator(get_path("atmprof_paranal.dat"))
    assert estim.find_shower_max_height(5*u.TeV, 1e5*u.m,70*u.deg).unit.is_equivalent(u.m), "return value has not proper unit"
  