"""Tests for MirrorDescription."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle

from ctapipe.instrument.optics import (
    MirrorDescription,
    MirrorFacetShape,
    OpticsDescription,
)


def _check_description(description):
    assert len(description.id) == 198
    assert description.id[0] == 198

    assert description.x[0].to_value(u.cm) == pytest.approx(461.99999999999994)
    assert description.y[0].to_value(u.cm) == pytest.approx(-1066.0799453238167)
    assert description.z[0].to_value(u.cm) == pytest.approx(120.53307587693142)
    assert description.surface_area[0].to_value(u.cm**2) == pytest.approx(
        19746.245231688983
    )

    assert description.nx[0] == pytest.approx(-0.08077963744975176)
    assert description.ny[0] == pytest.approx(0.18640162657079892)
    assert description.nz[0] == pytest.approx(0.9791471206030518)

    assert np.all(description.shape == MirrorFacetShape.HEXAGON)


def test_mirror_facets_description_from_ecsv(lst1_mirror_facets_path):
    description = OpticsDescription.from_mirror_description_table(
        lst1_mirror_facets_path
    )
    _check_description(description)


def test_get_facet_size_from_table(lst1_mirror_facets_path):
    """flat-to-flat distance for the (all-hexagon) LST1 facet table."""
    description = OpticsDescription.from_mirror_description_table(
        lst1_mirror_facets_path
    )
    size = description.get_facet_size()

    expected = np.sqrt(2 * description.surface_area[0] / np.sqrt(3))
    assert size[0].to_value(u.cm) == pytest.approx(expected.to_value(u.cm))
    assert size[0].to_value(u.cm) == pytest.approx(151.0)
    assert np.all(np.isfinite(size))


def test_get_facet_size_per_shape():
    """radius for CIRCLE, side for SQUARE, flat-to-flat for HEXAGON, nan for UNKNOWN."""
    side = 1.0
    hexagon_area = 3 * np.sqrt(3) / 2 * side**2

    description = MirrorDescription(
        id=np.arange(4),
        x=np.zeros(4) * u.m,
        y=np.zeros(4) * u.m,
        z=np.zeros(4) * u.m,
        nx=np.zeros(4),
        ny=np.zeros(4),
        nz=np.ones(4),
        surface_area=np.array([np.pi, side**2, hexagon_area, 1.0]) * u.m**2,
        mirror_shape=["CIRCLE", "SQUARE", "HEXAGON", "UNKNOWN"],
    )

    size = description.get_facet_size()

    assert size[0].to_value(u.m) == pytest.approx(1.0)  # radius
    assert size[1].to_value(u.m) == pytest.approx(1.0)  # side
    assert size[2].to_value(u.m) == pytest.approx(np.sqrt(3))  # flat-to-flat
    assert np.isnan(size[3].to_value(u.m))


@pytest.mark.parametrize(
    ("shape", "expected_deg"),
    [
        (MirrorFacetShape.UNKNOWN, 0),
        (MirrorFacetShape.CIRCLE, 0),
        (MirrorFacetShape.SQUARE, 45),
        (MirrorFacetShape.HEXAGON, 30),
        ("HEXAGON", 30),
    ],
)
def test_get_facet_rotation(shape, expected_deg):
    rotation = MirrorDescription.get_facet_rotation(shape)
    assert isinstance(rotation, Angle)
    assert rotation.isscalar
    assert rotation.to_value(u.deg) == expected_deg


def test_get_facet_rotation_array(lst1_mirror_facets_path):
    rotation = MirrorDescription.get_facet_rotation(
        ["UNKNOWN", "CIRCLE", "SQUARE", "HEXAGON"]
    )
    assert isinstance(rotation, Angle)
    np.testing.assert_array_equal(rotation.to_value(u.deg), [0, 0, 45, 30])

    description = OpticsDescription.from_mirror_description_table(
        lst1_mirror_facets_path
    )
    rotation = description.get_facet_rotation(description.shape)
    assert rotation.shape == (198,)
    assert np.all(rotation == Angle(30, u.deg))


@pytest.mark.parametrize("wrong_unit", [u.s, u.deg, u.kg, u.m**2])
def test_init_x_wrong_unit_type(wrong_unit):
    """x must have units of length."""
    with pytest.raises(u.UnitsError, match="Argument 'x'.*'length'"):
        MirrorDescription(
            id=np.arange(1),
            x=np.zeros(1) * wrong_unit,
            y=np.zeros(1) * u.m,
            z=np.zeros(1) * u.m,
            nx=np.zeros(1),
            ny=np.zeros(1),
            nz=np.ones(1),
            surface_area=np.ones(1) * u.m**2,
            mirror_shape=["HEXAGON"],
        )


def _optics_kwargs():
    from ctapipe.instrument.optics import ReflectorShape, SizeType

    return dict(
        name="test",
        size_type=SizeType.LST,
        reflector_shape=ReflectorShape.PARABOLIC,
        n_mirrors=1,
        n_mirror_tiles=198,
        mirror_area=u.Quantity(386, u.m**2),
        equivalent_focal_length=u.Quantity(28, u.m),
        effective_focal_length=u.Quantity(29.3, u.m),
    )


def _mirrors_to_table(mirrors):
    return OpticsDescription(**_optics_kwargs(), mirrors_description=mirrors).to_table()


def test_optics_to_table_roundtrip(lst1_mirror_facets_path):
    """The facets are written as the rows of the optics table."""
    mirrors = OpticsDescription.from_mirror_description_table(lst1_mirror_facets_path)
    optics = OpticsDescription(**_optics_kwargs(), mirrors_description=mirrors)
    table = optics.to_table()

    assert table.colnames == [
        "mirror_id",
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "surface",
        "shape",
    ]
    assert len(table) == 198
    assert table.meta["optics_name"] == "test"

    roundtripped = OpticsDescription.from_table(table)
    assert roundtripped == optics
    _check_description(roundtripped.mirrors_description)

    _check_description(OpticsDescription.from_mirror_description_table(table))


def test_optics_to_table_without_mirrors():
    optics = OpticsDescription(**_optics_kwargs())
    table = optics.to_table()

    assert len(table) == 0
    assert table.colnames == []

    roundtripped = OpticsDescription.from_table(table)
    assert roundtripped.mirrors_description is None
    assert roundtripped == optics


# optics metadata keys are longer than 8 characters -> FITS HIERARCH cards
@pytest.mark.filterwarnings("ignore::astropy.io.fits.verify.VerifyWarning")
def test_optics_to_table_roundtrip_via_file(tmp_path, lst1_mirror_facets_path):
    mirrors = OpticsDescription.from_mirror_description_table(lst1_mirror_facets_path)
    optics = OpticsDescription(**_optics_kwargs(), mirrors_description=mirrors)
    table = optics.to_table()

    ecsv_path = tmp_path / "roundtrip.ecsv"
    fits_path = tmp_path / "roundtrip.fits"
    table.write(ecsv_path)
    table.write(fits_path)

    for path in (ecsv_path, fits_path):
        roundtripped = OpticsDescription.from_table(path)
        assert roundtripped == optics
        _check_description(roundtripped.mirrors_description)


def test_optics_mirrors_description_attribute(lst1_mirror_facets_path):
    """OpticsDescription stores an optional MirrorDescription."""
    optics = OpticsDescription(**_optics_kwargs())
    assert optics.mirrors_description is None

    mirrors = OpticsDescription.from_mirror_description_table(lst1_mirror_facets_path)
    optics = OpticsDescription(**_optics_kwargs(), mirrors_description=mirrors)
    assert optics.mirrors_description is mirrors
    _check_description(optics.mirrors_description)


def test_mirror_description_hash_eq(lst1_mirror_facets_path):
    """Equal facets give equal hashes, also across units and table roundtrips."""
    description = OpticsDescription.from_mirror_description_table(
        lst1_mirror_facets_path
    )
    roundtripped = OpticsDescription.from_mirror_description_table(
        _mirrors_to_table(description)
    )
    assert description == roundtripped
    assert hash(description) == hash(roundtripped)

    in_meters = MirrorDescription(
        id=description.id,
        x=description.x.to(u.m),
        y=description.y.to(u.m),
        z=description.z.to(u.m),
        nx=description.nx,
        ny=description.ny,
        nz=description.nz,
        surface_area=description.surface_area.to(u.m**2),
        mirror_shape=description.shape,
    )
    assert description == in_meters

    shifted = MirrorDescription(
        id=description.id,
        x=description.x + 1 * u.cm,
        y=description.y,
        z=description.z,
        nx=description.nx,
        ny=description.ny,
        nz=description.nz,
        surface_area=description.surface_area,
        mirror_shape=description.shape,
    )
    assert description != shifted
    assert len({description, roundtripped, in_meters, shifted}) == 2


def test_optics_hash_eq_includes_mirrors_description(lst1_mirror_facets_path):
    mirrors = OpticsDescription.from_mirror_description_table(lst1_mirror_facets_path)
    without = OpticsDescription(**_optics_kwargs())
    with_mirrors = OpticsDescription(**_optics_kwargs(), mirrors_description=mirrors)
    with_same_mirrors = OpticsDescription(
        **_optics_kwargs(),
        mirrors_description=OpticsDescription.from_mirror_description_table(
            _mirrors_to_table(mirrors)
        ),
    )

    assert without == OpticsDescription(**_optics_kwargs())
    assert with_mirrors == with_same_mirrors
    assert hash(with_mirrors) == hash(with_same_mirrors)
    assert with_mirrors != without
