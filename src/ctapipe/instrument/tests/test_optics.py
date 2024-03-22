""" Tests for OpticsDescriptions"""

import pytest
from astropy import units as u

from ctapipe.instrument.optics import OpticsDescription, ReflectorShape, SizeType
from ctapipe.instrument.warnings import FromNameWarning


def test_guess_optics():
    """make sure we can guess an optics type from metadata"""
    from ctapipe.instrument import guess_telescope

    answer = guess_telescope(1855, 28.0 * u.m)

    assert answer.name == "LST"
    assert answer.n_mirrors == 1


def test_construct_optics():
    """create an OpticsDescription and make sure it
    fails if units are missing"""
    OpticsDescription(
        name="test",
        size_type=SizeType.LST,
        reflector_shape=ReflectorShape.PARABOLIC,
        n_mirrors=1,
        n_mirror_tiles=100,
        mirror_area=u.Quantity(550, u.m**2),
        equivalent_focal_length=u.Quantity(10, u.m),
        effective_focal_length=u.Quantity(11, u.m),
    )

    with pytest.raises(TypeError):
        # missing units
        OpticsDescription(
            name="test",
            size_type=SizeType.LST,
            reflector_shape=ReflectorShape.PARABOLIC,
            n_mirrors=1,
            n_mirror_tiles=100,
            mirror_area=550,
            equivalent_focal_length=10,
            effective_focal_length=11,
        )


@pytest.mark.parametrize(
    "optics_name,focal_length",
    zip(["LST", "MST", "ASTRI"], [28, 16, 2.15] * u.m),
)
def test_optics_from_name(optics_name, focal_length, svc_path):
    # test with file written by dump-instrument
    with pytest.warns(FromNameWarning):
        optics = OpticsDescription.from_name(optics_name)
    assert optics.name == optics_name
    assert u.isclose(optics.equivalent_focal_length, focal_length)
