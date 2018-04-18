from ctapipe.reco.shower_max import ShowerMaxEstimator
from astropy import units as u
from ctapipe.utils import get_dataset_path


def test_showermaxestimator(en=5 * u.TeV, h_first_int=10 * u.km, az=70 * u.deg):
    estim = ShowerMaxEstimator(atmosphere_profile_name='paranal')
    h_max = estim.find_shower_max_height(en, h_first_int, az)
    assert h_max.unit.is_equivalent(u.m), "return value has not proper unit"
    return h_max

if __name__ == "__main__":
    en, h_first_int, az = 5 * u.TeV, 10 * u.km, 70 * u.deg
    print("ShowerMaxEstimator unit test:")
    print("Energy = {}, hight of first interaction = {}, azimuth = {}"
          .format(en, h_first_int, az))
    print("h_max:", test_showermaxestimator())
