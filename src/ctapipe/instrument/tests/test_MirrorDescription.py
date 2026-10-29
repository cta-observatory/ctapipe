"""Tests for MirrorDescription."""

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from ctapipe.instrument.optics import (
    MirrorDescription,
    MirrorFacetShape,
    OpticsDescription,
)


def _check_description(description):
    assert len(description.facet_id) == 198
    assert np.all(description.mirror_id == 1)
    assert description.facet_id[0] == 1

    assert description.x[0].to_value(u.m) == pytest.approx(-3.08)
    assert description.y[0].to_value(u.m) == pytest.approx(-11.114)
    assert description.z[0].to_value(u.m) == pytest.approx(1.1875660357142859)

    assert description.nx[0] == pytest.approx(0.053869477715727455)
    assert description.ny[0] == pytest.approx(0.1943848621209724)
    assert description.nz[0] == pytest.approx(0.9794450493768628)

    assert description.rotation_angle[0].to_value(u.deg) == pytest.approx(0.0)
    assert description.diameter[0].to_value(u.m) == pytest.approx(1.51)
    assert description.focal_length[0].to_value(u.m) == pytest.approx(29.175)

    assert np.all(description.shape == MirrorFacetShape.HEXAGON)


def test_mirror_facets_description_from_ecsv(lst1_mirror_facets_path):
    description = OpticsDescription._mirror_description_from_table(
        Table.read(lst1_mirror_facets_path)
    )
    _check_description(description)


def test_optics_from_ecsv(lst1_mirror_facets_path):
    """The optics file can be read directly with OpticsDescription.from_table."""
    optics = OpticsDescription.from_table(lst1_mirror_facets_path)

    assert optics.name == "LSTN-01"
    assert optics.n_mirror_tiles == 198
    _check_description(optics.mirror_description)


@pytest.mark.parametrize(
    "path_fixture, n_facets",
    [
        ("lst1_mirror_facets_path", 198),
        ("MST_FlashCam_mirror_facets_path", 86),
        ("MSTN_NectarCam_mirror_facets_path", 86),
        ("SST_mirror_facets_path", 19),
    ],
)
def test_optics_from_ecsv_all_telescopes(request, path_fixture, n_facets):
    optics = OpticsDescription.from_table(request.getfixturevalue(path_fixture))
    mirrors = optics.mirror_description

    assert len(mirrors.facet_id) == n_facets == optics.n_mirror_tiles
    # facet ids are unique within each mirror
    assert len(set(zip(mirrors.mirror_id, mirrors.facet_id))) == n_facets
    assert mirrors == OpticsDescription.from_table(optics.to_table()).mirror_description


def test_sst_two_mirrors(SST_mirror_facets_path):
    """Hexagonal primary facets and a circular secondary, aspheric (nan focal length)."""
    mirrors = OpticsDescription.from_table(SST_mirror_facets_path).mirror_description

    primary = mirrors.mirror_id == 1
    secondary = mirrors.mirror_id == 2
    assert np.count_nonzero(primary) == 18
    assert np.count_nonzero(secondary) == 1

    assert np.all(mirrors.shape[primary] == MirrorFacetShape.HEXAGON)
    assert np.all(mirrors.shape[secondary] == MirrorFacetShape.CIRCLE)
    assert mirrors.diameter[secondary][0].to_value(u.m) == pytest.approx(1.8)
    assert np.all(mirrors.rotation_angle[primary] == 30 * u.deg)
    assert np.all(np.isnan(mirrors.focal_length))


def _mirror_kwargs(n=1, **kwargs):
    defaults = dict(
        mirror_id=np.ones(n, dtype=int),
        facet_id=np.arange(1, n + 1),
        x=np.zeros(n) * u.m,
        y=np.zeros(n) * u.m,
        z=np.zeros(n) * u.m,
        nx=np.zeros(n),
        ny=np.zeros(n),
        nz=np.ones(n),
        rotation_angle=np.zeros(n) * u.deg,
        diameter=np.ones(n) * u.m,
        mirror_shape=["HEXAGON"] * n,
        focal_length=np.full(n, 28.0) * u.m,
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.parametrize("shape", ["hexagon", "HEXAGON", "Hexagon"])
def test_init_shape_case_insensitive(shape):
    mirrors = MirrorDescription(**_mirror_kwargs(mirror_shape=[shape]))
    assert mirrors.shape[0] is MirrorFacetShape.HEXAGON


def test_init_invalid_shape():
    with pytest.raises(ValueError):
        MirrorDescription(**_mirror_kwargs(mirror_shape=["triangle"]))


@pytest.mark.parametrize("wrong_unit", [u.s, u.deg, u.kg, u.m**2])
def test_init_x_wrong_unit_type(wrong_unit):
    """x must have units of length."""
    with pytest.raises(u.UnitsError, match="Argument 'x'.*'length'"):
        MirrorDescription(**_mirror_kwargs(x=np.zeros(1) * wrong_unit))


@pytest.mark.parametrize(
    "argument, value, physical_type",
    [
        ("rotation_angle", np.zeros(1) * u.m, "angle"),
        ("diameter", np.ones(1) * u.deg, "length"),
        ("focal_length", np.ones(1) * u.m**2, "length"),
    ],
)
def test_init_wrong_unit_type(argument, value, physical_type):
    with pytest.raises(u.UnitsError, match=f"Argument '{argument}'.*'{physical_type}'"):
        MirrorDescription(**_mirror_kwargs(**{argument: value}))


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
    return OpticsDescription(**_optics_kwargs(), mirror_description=mirrors).to_table()


def test_optics_to_table_roundtrip(lst1_mirror_facets_path):
    """The facets are written as the rows of the optics table."""
    mirrors = OpticsDescription._mirror_description_from_table(
        Table.read(lst1_mirror_facets_path)
    )
    optics = OpticsDescription(**_optics_kwargs(), mirror_description=mirrors)
    table = optics.to_table()

    assert table.colnames == [
        "mirror_id",
        "facet_id",
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "rotation_angle",
        "diameter",
        "shape",
        "focal_length",
    ]
    assert len(table) == 198
    assert table.meta["optics_name"] == "test"

    roundtripped = OpticsDescription.from_table(table)
    assert roundtripped == optics
    _check_description(roundtripped.mirror_description)

    _check_description(OpticsDescription._mirror_description_from_table(table))


def test_optics_to_table_without_mirrors():
    optics = OpticsDescription(**_optics_kwargs())
    table = optics.to_table()

    assert len(table) == 0
    assert table.colnames == []

    roundtripped = OpticsDescription.from_table(table)
    assert roundtripped.mirror_description is None
    assert roundtripped == optics


# optics metadata keys are longer than 8 characters -> FITS HIERARCH cards
@pytest.mark.filterwarnings("ignore::astropy.io.fits.verify.VerifyWarning")
def test_optics_to_table_roundtrip_via_file(tmp_path, lst1_mirror_facets_path):
    mirrors = OpticsDescription._mirror_description_from_table(
        Table.read(lst1_mirror_facets_path)
    )
    optics = OpticsDescription(**_optics_kwargs(), mirror_description=mirrors)
    table = optics.to_table()

    ecsv_path = tmp_path / "roundtrip.ecsv"
    fits_path = tmp_path / "roundtrip.fits"
    table.write(ecsv_path)
    table.write(fits_path)

    for path in (ecsv_path, fits_path):
        roundtripped = OpticsDescription.from_table(path)
        assert roundtripped == optics
        _check_description(roundtripped.mirror_description)


def test_optics_mirror_description_attribute(lst1_mirror_facets_path):
    """OpticsDescription stores an optional MirrorDescription."""
    optics = OpticsDescription(**_optics_kwargs())
    assert optics.mirror_description is None

    mirrors = OpticsDescription._mirror_description_from_table(
        Table.read(lst1_mirror_facets_path)
    )
    optics = OpticsDescription(**_optics_kwargs(), mirror_description=mirrors)
    assert optics.mirror_description is mirrors
    _check_description(optics.mirror_description)


def test_mirror_description_hash_eq(lst1_mirror_facets_path):
    """Equal facets give equal hashes, also across units and table roundtrips."""
    description = OpticsDescription._mirror_description_from_table(
        Table.read(lst1_mirror_facets_path)
    )
    roundtripped = OpticsDescription._mirror_description_from_table(
        _mirrors_to_table(description)
    )
    assert description == roundtripped
    assert hash(description) == hash(roundtripped)

    in_cm = MirrorDescription(
        mirror_id=description.mirror_id,
        facet_id=description.facet_id,
        x=description.x.to(u.cm),
        y=description.y.to(u.cm),
        z=description.z.to(u.cm),
        nx=description.nx,
        ny=description.ny,
        nz=description.nz,
        rotation_angle=description.rotation_angle.to(u.rad),
        diameter=description.diameter.to(u.cm),
        mirror_shape=description.shape,
        focal_length=description.focal_length.to(u.cm),
    )
    assert description == in_cm

    shifted = MirrorDescription(
        mirror_id=description.mirror_id,
        facet_id=description.facet_id,
        x=description.x + 1 * u.cm,
        y=description.y,
        z=description.z,
        nx=description.nx,
        ny=description.ny,
        nz=description.nz,
        rotation_angle=description.rotation_angle,
        diameter=description.diameter,
        mirror_shape=description.shape,
        focal_length=description.focal_length,
    )
    assert description != shifted
    assert len({description, roundtripped, in_cm, shifted}) == 2


def test_optics_hash_eq_includes_mirror_description(lst1_mirror_facets_path):
    mirrors = OpticsDescription._mirror_description_from_table(
        Table.read(lst1_mirror_facets_path)
    )
    without = OpticsDescription(**_optics_kwargs())
    with_mirrors = OpticsDescription(**_optics_kwargs(), mirror_description=mirrors)
    with_same_mirrors = OpticsDescription(
        **_optics_kwargs(),
        mirror_description=OpticsDescription._mirror_description_from_table(
            _mirrors_to_table(mirrors)
        ),
    )

    assert without == OpticsDescription(**_optics_kwargs())
    assert with_mirrors == with_same_mirrors
    assert hash(with_mirrors) == hash(with_same_mirrors)
    assert with_mirrors != without
