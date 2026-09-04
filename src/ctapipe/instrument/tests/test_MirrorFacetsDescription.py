"""Tests for MirrorFacetsDescription."""

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from ctapipe.instrument.optics import MirrorFacetsDescription, MirrorFacetShape

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


def test_create_patches_from_table():
    """create_patches on the real (all-hexagon) LST1 facet table."""
    pytest.importorskip("matplotlib")
    from matplotlib.patches import RegularPolygon

    description = MirrorFacetsDescription.from_table(ECSV_PATH)
    size = description.get_facet_size()

    patches = MirrorFacetsDescription.create_patches(
        shape=description.shape[0],
        facet_x=description.x.to_value(u.m),
        facet_y=description.y.to_value(u.m),
        facet_size=size.to_value(u.m),
    )

    assert len(patches) == 198
    assert all(isinstance(patch, RegularPolygon) for patch in patches)

    # hexagon patch radius is the outer-circle radius, i.e.
    # flat-to-flat distance / sqrt(3)
    assert patches[0].radius == pytest.approx(size[0].to_value(u.m) / np.sqrt(3))
    assert patches[0].xy == pytest.approx(
        (description.x[0].to_value(u.m), description.y[0].to_value(u.m))
    )


def test_create_patches_per_shape():
    """create_patches dispatches correctly for CIRCLE, SQUARE and HEXAGON."""
    pytest.importorskip("matplotlib")
    from matplotlib.patches import Circle, RegularPolygon

    from ctapipe.instrument.optics import MirrorFacetShape

    # circle: facet_size is already the radius
    (circle,) = MirrorFacetsDescription.create_patches(
        MirrorFacetShape.CIRCLE, [0.0], [0.0], [2.0]
    )
    assert isinstance(circle, Circle)
    assert circle.radius == pytest.approx(2.0)

    # square: facet_size is the side length -> outer radius = side / sqrt(2)
    (square,) = MirrorFacetsDescription.create_patches(
        MirrorFacetShape.SQUARE, [0.0], [0.0], [1.0]
    )
    assert isinstance(square, RegularPolygon)
    assert square.radius == pytest.approx(1.0 / np.sqrt(2))

    # hexagon: facet_size is flat-to-flat -> outer radius = flat_to_flat / sqrt(3)
    (hexagon,) = MirrorFacetsDescription.create_patches(
        MirrorFacetShape.HEXAGON, [0.0], [0.0], [1.0]
    )
    assert isinstance(hexagon, RegularPolygon)
    assert hexagon.radius == pytest.approx(1.0 / np.sqrt(3))

    with pytest.raises(ValueError, match="Unsupported mirror facet shape"):
        MirrorFacetsDescription.create_patches(
            MirrorFacetShape.UNKNOWN, [0.0], [0.0], [1.0]
        )
