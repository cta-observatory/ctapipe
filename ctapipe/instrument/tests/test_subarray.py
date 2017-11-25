from ctapipe.instrument import (
    SubarrayDescription,
    TelescopeDescription,
)
import numpy as np
from astropy import units as u


def test_subarray_description():

    pos = {}
    tel = {}
    foclen = 16 * u.m
    pix_x = np.arange(1764, dtype=np.float) * u.m
    pix_y = np.arange(1764, dtype=np.float) * u.m

    for ii in range(10):
        tel[ii] = TelescopeDescription.guess(pix_x, pix_y, foclen)
        pos[ii] = np.random.uniform(-100, 100, size=2) * u.m

    sub = SubarrayDescription("test array",
                              tel_positions=pos,
                              tel_descriptions=tel)

    sub.info()

    assert sub.num_tels == 10
    assert sub.tel[0].camera is not None
    assert len(sub.to_table()) == 10

    subsub = sub.select_subarray("newsub", [1, 2, 3, 4])
    assert subsub.num_tels == 4
    assert set(subsub.tels.keys()) == {1, 2, 3, 4}
