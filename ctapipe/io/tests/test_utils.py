'''This test, tests a function, which is soon to be removed anyway...
'''

from ctapipe.io.utils import is_fits_in_header
from ctapipe.utils import get_dataset_path

a_fits_file = get_dataset_path('LSTCam.camgeom.fits.gz')
not_a_fits_file = get_dataset_path("gamma_test.simtel.gz")


def test_is_fits_in_header():
    assert is_fits_in_header(a_fits_file)
    assert not is_fits_in_header(not_a_fits_file)
