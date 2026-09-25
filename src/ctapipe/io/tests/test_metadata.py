"""
Test CTA Reference metadata functionality
"""

import json
import uuid

import ctao_datamodel as dm
import ctao_datamodel.models.dataproducts as dp
import pytest
import tables
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from ctao_datamodel.models.common import SiteID
from pydantic import ValidationError

from ctapipe.core.provenance import Provenance
from ctapipe.io import metadata as meta
from ctapipe.io.datalevels import DataLevel


@pytest.fixture()
def reference():
    prov = Provenance()
    prov.start_activity("test")
    prov.finish_activity()
    prov_activity = prov.finished_activities[0]

    reference = meta.Reference(
        contact=meta.Contact(
            name="Somebody",
            email="a@b.com",
            organization="CTA Consortium",
        ),
        product=meta.Product(
            description="An Amazing Product",
            creation_time="2020-10-11 15:23:31",
            data_category="Sim",
            data_levels=["DL1_IMAGES", "DL1_PARAMETERS"],
            data_association="Subarray",
            data_model_name="Unofficial DL1",
            data_model_version="1.0",
            data_model_url="https://example.org",
            format="hdf5",
        ),
        process=meta.Process(type_="Simulation", subtype="Prod3b", id_="423442"),
        activity=meta.Activity.from_provenance(prov_activity.provenance),
        instrument=meta.Instrument(
            site="CTA-North",
            class_="Array",
            type_="Layout H1B",
            version="1.0",
            id_="threshold",
        ),
    )
    return reference


@pytest.fixture()
def ctao_product():
    return dp.Product(
        description="ctapipe test product",
        creation_time=Time("2026-09-21T12:34:56"),
        data=dp.ProductType(
            level=dp.DataLevel.DL1,
            division=dp.DataDivision.EVENT,
            association=dp.DataAssociation.SUBARRAY,
            type=dp.DataType.OBSERVATION,
        ),
        instance=dp.InstanceIdentifier(
            id=uuid.UUID("f08d6e7c-166e-4da2-b850-85a7a454c1e6"),
            obs_id=42,
            category=dp.DataProcessingCategory.B,
        ),
        curation=dp.Curation(release="test"),
        model=dp.DataModel(
            name="ctapipe",
            version="v7.6.0",
            url="https://example.org/model",
        ),
        contact=dp.Contact(
            name="Test User",
            organization="CTAO",
            email="test@example.org",
        ),
        activity=dp.Activity(
            process=dp.ObservatoryProcess.DATA_PROCESSING,
            name="ctapipe-process",
            id=uuid.UUID("fb9d405f-9163-4296-ac2d-f708e6a6b113"),
            start=Time("2026-09-21T12:00:00"),
            configuration_id="test-config",
            software=dp.Software(
                name="ctapipe",
                version="0.32",
                url="https://ctapipe.readthedocs.io",
            ),
        ),
    )


@pytest.fixture()
def legacy_file(tmp_path, reference):
    path = tmp_path / "legacy.h5"
    with tables.open_file(path, mode="w") as h5file:
        meta.write_to_hdf5(reference.to_dict(), h5file)
    return path


def test_to_dict(reference):
    """Test for Reference.to_dict"""
    ref_dict = reference.to_dict()
    assert ref_dict["CTA PRODUCT FORMAT"] == "hdf5"
    assert ref_dict["CTA PRODUCT DATA LEVELS"] == "DL1_IMAGES,DL1_PARAMETERS"
    assert str(uuid.UUID(ref_dict["CTA PRODUCT ID"])) == ref_dict["CTA PRODUCT ID"]


def test_from_dict(reference):
    as_dict = reference.to_dict()
    back = meta.Reference.from_dict(as_dict)
    assert back.to_dict() == as_dict


@pytest.mark.parametrize("format", ("fits", "fits.gz"))
def test_reference_metadata_fits(tmp_path, format, reference):
    """Test for writing reference metadata"""
    path = tmp_path / f"test.{format}"

    hdul = fits.HDUList(fits.PrimaryHDU())
    hdul[0].header.update(reference.to_dict(fits=True))
    hdul.writeto(path)

    back = meta.read_reference_metadata(path)
    assert back.to_dict() == reference.to_dict()


def test_reference_metadata_h5(tmp_path, reference):
    path = tmp_path / "test.h5"

    with tables.open_file(path, "w") as f:
        meta.write_to_hdf5(reference.to_dict(), f)

    back = meta.read_reference_metadata(path)
    assert back.to_dict() == reference.to_dict()


def test_reference_metadata_ecsv(tmp_path, reference):
    path = tmp_path / "test.ecsv"

    t = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
    t.meta.update(reference.to_dict())
    t.write(path)

    back = meta.read_reference_metadata(path)
    assert back.to_dict() == reference.to_dict()


def test_read_hdf5_metadata(tmp_path):
    # Testing one can read both a path as well as a PyTables file object
    filename = tmp_path / "test.h5"
    metadata_in = {"SOFTWARE": "ctapipe", "FOO": "BAR"}
    metadata_path = "/node/subnode"
    with tables.open_file(filename, mode="w") as h5file:
        h5file.create_group(where="/node", name="subnode", createparents=True)
        meta.write_to_hdf5(metadata_in, h5file, path=metadata_path)

    metadata_out = meta._read_hdf5_metadata(filename, path=metadata_path)
    assert metadata_out == metadata_in

    with tables.open_file(filename, "r") as file:
        metadata_out = meta._read_hdf5_metadata(file, path=metadata_path)

    assert metadata_out == metadata_in


def _write_current_metadata(path, product, file_format):
    flat = dm.flatten_model_instance(product, parent_key="CTAO")

    if file_format == "hdf5":
        with tables.open_file(path, mode="w") as h5file:
            meta.write_product_metadata(product, h5file)
    elif file_format == "fits":
        header = fits.Header()
        header.update(flat)
        fits.PrimaryHDU(header=header).writeto(path)
    elif file_format == "ecsv":
        Table({"value": [1]}, meta=flat).write(path)
    elif file_format == "json":
        path.write_text(json.dumps({"metadata": flat}))


@pytest.mark.parametrize("file_format", ["hdf5", "fits", "ecsv", "json"])
def test_read_current_metadata_supported_formats(tmp_path, ctao_product, file_format):
    path = tmp_path / f"product.{file_format}"
    _write_current_metadata(path, ctao_product, file_format)

    assert meta.read_ctao_metadata(path) == ctao_product


def test_read_current_metadata_from_open_hdf5(tmp_path, ctao_product):
    path = tmp_path / "product.h5"
    _write_current_metadata(path, ctao_product, "hdf5")

    with tables.open_file(path) as h5file:
        assert meta.read_ctao_metadata(h5file) == ctao_product
        assert h5file.isopen


def test_read_metadata_ignores_unrelated_attributes(tmp_path, ctao_product):
    path = tmp_path / "product.h5"
    with tables.open_file(path, mode="w") as h5file:
        meta.write_product_metadata(ctao_product, h5file)
        h5file.root._v_attrs["unrelated"] = "metadata"

    assert meta.read_ctao_metadata(path) == ctao_product


def test_read_legacy_metadata_derives_product_type(legacy_file):
    with pytest.warns(meta.LegacyMetadataWarning, match="Legacy"):
        product = meta.read_ctao_metadata(legacy_file)

    assert product.description == "An Amazing Product"
    assert product.data == dp.ProductType(
        level=dp.DataLevel.DL1,
        division=dp.DataDivision.EVENT,
        association=dp.DataAssociation.SUBARRAY,
        type=dp.DataType.OBSERVATION_SIM,
    )
    assert product.contact.email == "a@b.com"


def test_legacy_metadata_preserves_valid_ids(reference):
    product = meta._legacy_reference_to_product(
        reference,
        meta._legacy_product_type(None, reference),
        contact_fallback=dp.Contact(
            name="Fallback", organization="CTAO", email="fallback@example.org"
        ),
    )

    assert product.instance.id == uuid.UUID(reference.product.id_)
    assert product.activity.id == uuid.UUID(reference.activity.id_)


@pytest.mark.parametrize(("field", "value"), [("product", "invalid"), ("activity", "")])
def test_legacy_metadata_replaces_invalid_ids(reference, field, value):
    if field == "product":
        reference.product.id_ = value
    else:
        reference.activity.id_ = value

    with pytest.warns(meta.LegacyMetadataWarning, match=f"invalid {field} id"):
        product = meta._legacy_reference_to_product(
            reference,
            meta._legacy_product_type(None, reference),
            contact_fallback=dp.Contact(
                name="Fallback", organization="CTAO", email="fallback@example.org"
            ),
        )

    assert isinstance(product.instance.id, uuid.UUID)
    assert isinstance(product.activity.id, uuid.UUID)


def test_read_legacy_metadata_from_open_hdf5(legacy_file):
    with tables.open_file(legacy_file) as h5file:
        with pytest.warns(meta.LegacyMetadataWarning):
            product = meta.read_ctao_metadata(h5file)
        assert product.data.type is dp.DataType.OBSERVATION_SIM
        assert h5file.isopen


@pytest.mark.parametrize(
    ("levels", "expected_level", "expected_sublevel"),
    [
        ([DataLevel.R0], dp.DataLevel.R0, None),
        ([DataLevel.DL1_IMAGES], dp.DataLevel.DL1, dp.ProcessingSublevel.IMAGES),
        (
            [DataLevel.DL1_PARAMETERS],
            dp.DataLevel.DL1,
            dp.ProcessingSublevel.PARAMETERS,
        ),
        (
            [DataLevel.DL1_IMAGES, DataLevel.DL1_PARAMETERS],
            dp.DataLevel.DL1,
            None,
        ),
        ([DataLevel.R1, DataLevel.DL2], dp.DataLevel.DL2, None),
    ],
)
def test_legacy_data_level_and_sublevel_mapping(
    reference, levels, expected_level, expected_sublevel
):
    reference.product.data_levels = levels
    product_type = dp.ProductType(
        level=expected_level,
        division=dp.DataDivision.EVENT,
        association=dp.DataAssociation.SUBARRAY,
        type=dp.DataType.OBSERVATION_SIM,
    )

    product = meta._legacy_reference_to_product(
        reference,
        product_type,
        contact_fallback=dp.Contact(
            name="Fallback", organization="CTAO", email="fallback@example.org"
        ),
    )
    assert product.data.level is expected_level
    assert product.instance.sublevel_id is expected_sublevel


def test_legacy_product_type_rejects_missing_level_and_unknown_association(reference):
    reference.product.data_levels = []
    with pytest.raises(ValueError, match="At least one data level"):
        meta._legacy_product_type(None, reference)

    reference.product.data_levels = [DataLevel.DL1_IMAGES]
    reference.product.data_association = "Other"
    with pytest.raises(ValueError, match="Unsupported legacy data association"):
        meta._legacy_product_type(None, reference)


@pytest.mark.parametrize(
    ("instrument_class", "instrument_id", "expected"),
    [
        ("Telescope", "23", {"ae_class": dp.ArrayElementClass.TEL, "ae_id": 23}),
        ("Subarray", "17", {"subarray_id": 17}),
        ("Array", "not-an-integer", {}),
    ],
)
def test_legacy_instrument_mapping(
    reference, instrument_class, instrument_id, expected
):
    reference.instrument.site = "South"
    reference.instrument.class_ = instrument_class
    reference.instrument.id_ = instrument_id
    product_type = meta._legacy_product_type(None, reference)

    product = meta._legacy_reference_to_product(
        reference,
        product_type,
        contact_fallback=dp.Contact(
            name="Fallback", organization="CTAO", email="fallback@example.org"
        ),
    )

    assert product.instance.site_id is SiteID.CTAO_SOUTH
    for name, value in expected.items():
        assert getattr(product.instance, name) == value


def test_legacy_missing_optional_values_and_contact_fallback(reference):
    reference.product.data_model_url = " unspecified "
    reference.contact.name = "unknown"
    reference.contact.organization = ""
    reference.contact.email = "not-an-email"
    fallback = dp.Contact(
        name="Fallback", organization="CTAO", email="fallback@example.org"
    )

    with pytest.warns(meta.LegacyMetadataWarning, match="invalid contact"):
        product = meta._legacy_reference_to_product(
            reference,
            meta._legacy_product_type(None, reference),
            contact_fallback=fallback,
        )

    assert product.contact == fallback
    assert product.model.url is None


@pytest.mark.parametrize(
    ("process_type", "category", "has_simulation_group", "expected"),
    [
        ("Simulation", "Other", False, dp.DataType.OBSERVATION_SIM),
        ("Observation", "Sim", True, dp.DataType.OBSERVATION),
        ("Other", "Sim", False, dp.DataType.OBSERVATION_SIM),
        ("Other", "Other", True, dp.DataType.OBSERVATION_SIM),
        ("Other", "Other", False, dp.DataType.OBSERVATION),
    ],
)
def test_legacy_data_type_fallbacks(
    tmp_path, reference, process_type, category, has_simulation_group, expected
):
    reference.process.type_ = process_type
    reference.product.data_category = category
    path = tmp_path / "legacy.h5"
    with tables.open_file(path, mode="w") as h5file:
        if has_simulation_group:
            h5file.create_group("/configuration", "simulation", createparents=True)

    with tables.open_file(path) as h5file:
        assert meta.to_ctao_data_type(reference, h5file) is expected


def test_invalid_and_missing_metadata(tmp_path, ctao_product):
    unsupported = tmp_path / "unsupported.h5"
    with tables.open_file(unsupported, mode="w"):
        pass

    with pytest.raises(ValueError, match="Unsupported metadata format"):
        meta.read_ctao_metadata(unsupported)

    invalid = tmp_path / "invalid.h5"
    with tables.open_file(invalid, mode="w") as h5file:
        meta.write_product_metadata(ctao_product, h5file)
        h5file.root._v_attrs["CTAO.contact.email"] = "invalid"

    with pytest.raises(ValidationError):
        meta.read_ctao_metadata(invalid)


def test_write_product_metadata_removes_only_legacy(tmp_path, ctao_product, reference):
    path = tmp_path / "product.h5"
    with tables.open_file(path, mode="w") as h5file:
        meta.write_to_hdf5(reference.to_dict(), h5file)
        h5file.root._v_attrs["CONTEXT custom"] = "keep"
        meta.write_product_metadata(ctao_product, h5file, remove_legacy=True)
        attributes = meta._read_hdf5_metadata(h5file)

    assert not any(name.startswith("CTA ") for name in attributes)
    assert attributes["CONTEXT custom"] == "keep"
    assert attributes["CTAO.data.level"] == "DL1"


def test_reprs(reference):
    assert isinstance(repr(reference), str)
    assert isinstance(repr(reference.activity), str)
    assert isinstance(repr(reference.product), str)
    assert isinstance(repr(reference.contact), str)
    assert isinstance(repr(reference.instrument), str)
    assert isinstance(repr(reference.process), str)
