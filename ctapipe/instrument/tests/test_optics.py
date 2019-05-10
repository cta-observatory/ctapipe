from ctapipe.instrument.optics import OpticsDescription
from astropy import units as u
import pytest


def test_guess_optics():
    from ctapipe.instrument import guess_telescope

    answer = guess_telescope(1855, 28.0 * u.m)

    od = OpticsDescription.from_name(answer.name)

    assert od.equivalent_focal_length.to_value(u.m) == 28
    assert od.num_mirrors == 1


def test_construct_optics():
    OpticsDescription(
        name="test",
        num_mirrors=1,
        num_mirror_tiles=100,
        mirror_area=u.Quantity(550, u.m ** 2),
        equivalent_focal_length=u.Quantity(10, u.m),
    )

    with pytest.raises(TypeError):
        OpticsDescription(
            name="test",
            num_mirrors=1,
            num_mirror_tiles=100,
            mirror_area=550,
            equivalent_focal_length=10,
        )


@pytest.mark.parametrize("optics_name", OpticsDescription.get_known_optics_names())
def test_optics_from_name(optics_name):
    optics = OpticsDescription.from_name(optics_name)
    assert optics.equivalent_focal_length > 0
    # make sure the string rep gives back the name:
    assert str(optics) == optics_name
