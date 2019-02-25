import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from ctapipe.instrument import (
    SubarrayDescription,
    TelescopeDescription,
)


def test_subarray_description():
    pos = {}
    tel = {}
    n_tels = 10

    for tel_id in range(1, n_tels + 1):
        tel[tel_id] = TelescopeDescription.from_name(
            optics_name="MST",
            camera_name="NectarCam",
        )
        pos[tel_id] = np.random.uniform(-100, 100, size=3) * u.m

    sub = SubarrayDescription(
        "test array",
        tel_positions=pos,
        tel_descriptions=tel
    )

    assert len(sub.telescope_types) == 1

    assert str(sub) == "test array"
    assert sub.num_tels == n_tels
    assert len(sub.tel_ids) == n_tels
    assert sub.tel_ids[0] == 1
    assert sub.tel[1].camera is not None
    assert 0 not in sub.tel  # check that there is no tel 0 (1 is first above)
    assert len(sub.to_table()) == n_tels
    assert len(sub.camera_types) == 1  # only 1 camera type
    assert sub.camera_types[0] == 'NectarCam'
    assert sub.optics_types[0].equivalent_focal_length.to_value(u.m) == 16.0
    assert sub.telescope_types[0] == 'MST:NectarCam'
    assert sub.tel_coords
    assert isinstance(sub.tel_coords, SkyCoord)
    assert len(sub.tel_coords) == n_tels

    subsub = sub.select_subarray("newsub", [2, 3, 4, 6])
    assert subsub.num_tels == 4
    assert set(subsub.tels.keys()) == {2, 3, 4, 6}
    assert subsub.tel_indices[6] == 3
    assert subsub.tel_ids[3] == 6

    assert len(sub.to_table(kind='optics')) == 1


if __name__ == '__main__':
    test_subarray_description()
