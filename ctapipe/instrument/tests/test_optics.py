from ctapipe.instrument.optics import OpticsDescription
from astropy import units as u
import pytest


def test_guess_optics():
    from ctapipe.instrument import guess_telescope
    answer = guess_telescope(1855, 28.0 * u.m)
    od = OpticsDescription.from_name(answer.telescope_name)
    od.info()

    assert od.tel_type == 'LST'
    assert od.tel_subtype == ''
    assert od.mirror_type == 'DC'


def test_construct_optics():
    with pytest.raises(ValueError):
        OpticsDescription(
            mirror_type="DC",
            tel_type="bad",  # bad value
            tel_subtype="1M",
            equivalent_focal_length=10 * u.m,
        )

    with pytest.raises(ValueError):
        OpticsDescription(
            mirror_type="bad",  # bad value
            tel_type="MST",
            tel_subtype="1M",
            equivalent_focal_length=10 * u.m,
        )
