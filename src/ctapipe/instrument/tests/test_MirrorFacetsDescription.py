"""Tests for MirrorFacetsDescription."""

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from ctapipe.instrument.optics import MirrorFacetShape, MirrorFacetsDescription

# Real mirror facet tables shipped alongside the repository for local testing.
DATA_DIR = Path(__file__).resolve().parents[4] / "tmp_files"
ECSV_PATH = DATA_DIR / "mirror_CTA-N-LST1_v2019-03-31.ecsv"
FITS_PATH = DATA_DIR / "mirror_CTA-N-LST1_v2019-03-31.fits"

pytestmark = pytest.mark.skipif(
    not (ECSV_PATH.exists() and FITS_PATH.exists()),
    reason="tmp_files/ mirror facet test data not available",
)


def _check_description(description):
    assert len(description.id) == 198
    assert description.id[0] == 198

    assert description.x[0].to_value(u.cm) == pytest.approx(461.99999999999994)
    assert description.y[0].to_value(u.cm) == pytest.approx(-1066.0799453238167)
    assert description.z[0].to_value(u.cm) == pytest.approx(120.53307587693142)
    assert description.surface[0].to_value(u.cm**2) == pytest.approx(19746.245231688983)

    assert description.nx[0] == pytest.approx(-0.08077963744975176)
    assert description.ny[0] == pytest.approx(0.18640162657079892)
    assert description.nz[0] == pytest.approx(0.9791471206030518)

    assert np.all(description.shape == MirrorFacetShape.HEXAGON)


def test_mirror_facets_description_from_ecsv():
    description = MirrorFacetsDescription.from_table(ECSV_PATH)
    _check_description(description)


def test_mirror_facets_description_from_fits():
    description = MirrorFacetsDescription.from_table(FITS_PATH)
    _check_description(description)


def test_get_facet_size_from_table():
    """flat-to-flat distance for the (all-hexagon) LST1 facet table."""
    description = MirrorFacetsDescription.from_table(ECSV_PATH)
    size = description.get_facet_size()

    expected = np.sqrt(2 * description.surface[0] / np.sqrt(3))
    assert size[0].to_value(u.cm) == pytest.approx(expected.to_value(u.cm))
    assert size[0].to_value(u.cm) == pytest.approx(151.0)
    assert np.all(np.isfinite(size))


def test_get_facet_size_per_shape():
    """radius for CIRCLE, side for SQUARE, flat-to-flat for HEXAGON, nan for UNKNOWN."""
    side = 1.0
    hexagon_area = 3 * np.sqrt(3) / 2 * side**2

    description = MirrorFacetsDescription(
        id=np.arange(4),
        x=np.zeros(4) * u.m,
        y=np.zeros(4) * u.m,
        z=np.zeros(4) * u.m,
        nx=np.zeros(4),
        ny=np.zeros(4),
        nz=np.ones(4),
        surface=np.array([np.pi, side**2, hexagon_area, 1.0]) * u.m**2,
        mirror_shape=["CIRCLE", "SQUARE", "HEXAGON", "UNKNOWN"],
    )

    size = description.get_facet_size()

    assert size[0].to_value(u.m) == pytest.approx(1.0)  # radius
    assert size[1].to_value(u.m) == pytest.approx(1.0)  # side
    assert size[2].to_value(u.m) == pytest.approx(np.sqrt(3))  # flat-to-flat
    assert np.isnan(size[3].to_value(u.m))
