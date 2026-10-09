"""Tests for MirrorDescription."""

import astropy.units as u
import numpy as np
import pytest

from ctapipe.instrument.optics import MirrorDescription, MirrorFacetShape


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
    description = MirrorDescription.from_table(lst1_mirror_facets_path)
    _check_description(description)


def test_get_facet_size_from_table(lst1_mirror_facets_path):
    """flat-to-flat distance for the (all-hexagon) LST1 facet table."""
    description = MirrorDescription.from_table(lst1_mirror_facets_path)
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


def test_to_table_roundtrip(lst1_mirror_facets_path):
    description = MirrorDescription.from_table(lst1_mirror_facets_path)
    table = description.to_table()

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
    assert table.meta["EXTNAME"] == "MIRRORS"

    roundtripped = MirrorDescription.from_table(table)
    _check_description(roundtripped)
    np.testing.assert_array_equal(roundtripped.id, description.id)


def test_to_table_roundtrip_via_file(tmp_path, lst1_mirror_facets_path):
    description = MirrorDescription.from_table(lst1_mirror_facets_path)
    table = description.to_table()

    ecsv_path = tmp_path / "roundtrip.ecsv"
    fits_path = tmp_path / "roundtrip.fits"
    table.write(ecsv_path)
    table.write(fits_path)

    for path in (ecsv_path, fits_path):
        roundtripped = MirrorDescription.from_table(path)
        _check_description(roundtripped)
