import numpy as np
from astropy import units as u

from ctapipe.instrument import TelescopeDescription


def test_telescope_description():
    # setup a dummy telescope that look like an MST with FlashCam
    foclen = 16 * u.m
    pix_x = np.arange(1764, dtype=np.float) * u.m
    pix_y = np.arange(1764, dtype=np.float) * u.m

    tel = TelescopeDescription.guess(pix_x, pix_y, foclen)

    assert tel.camera.cam_id == 'FlashCam'
    assert tel.optics.tel_type == 'MST'
    assert str(tel) == 'MST:FlashCam'
