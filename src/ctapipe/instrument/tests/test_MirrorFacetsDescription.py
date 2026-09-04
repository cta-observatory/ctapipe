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
    assert description.surface[0].to_value(u.cm**2) == pytest.approx(
        19746.245231688983
    )

    assert description.nx[0] == pytest.approx(-0.08077963744975176)
    assert description.ny[0] == pytest.approx(0.18640162657079892)
    assert description.nz[0] == pytest.approx(0.9791471206030518)

    assert np.all(description.shape == MirrorFacetShape.HEXAGON)


def test_mirror_facets_description_from_ecsv():
    description = MirrorFacetsDescription.from_table(ECSV_PATH)
    _check_description(description)
    print(ECSV_PATH)
    print(description)

def test_mirror_facets_description_from_fits():
    description = MirrorFacetsDescription.from_table(FITS_PATH)
    _check_description(description)
    print(FITS_PATH)
    print(description)
