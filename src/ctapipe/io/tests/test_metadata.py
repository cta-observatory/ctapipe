"""
Test CTA Reference metadata functionality
"""

import uuid

import pytest
import tables
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from ctao_datamodel.models import dataproducts as dp

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
            level="DL1",
            division="Event",
            association="Subarray",
            type="Observation",
        ),
        instance=dp.InstanceIdentifier(obs_id=42, category="B"),
        curation=dp.Curation(release="test"),
        model=dp.DataModel(
            name="CTAO",
            version="1.0",
            url="https://example.org/model",
        ),
        contact=dp.Contact(
            name="Test User",
            organization="CTAO",
            email="test@example.org",
        ),
        activity=dp.Activity(
            process="data_processing",
            name="ctapipe-process",
            id=uuid.uuid4(),
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
def ctao_product_type():
    return dp.ProductType(
        level="DL1",
        division="Event",
        association="Subarray",
        type="ObservationSim",
    )


@pytest.fixture()
def legacy_reference_file(tmp_path, reference):
    filename = tmp_path / "legacy_reference.h5"

    with tables.open_file(filename, mode="w") as h5file:
        meta.write_to_hdf5(reference.to_dict(), h5file)

    return filename


@pytest.fixture()
def reference_file(tmp_path, ctao_product):
    filename = tmp_path / "reference.h5"

    with tables.open_file(filename, mode="w") as h5file:
        meta.write_product_metadata(ctao_product, h5file)

    return filename


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

    metadata_out = meta.read_hdf5_metadata(filename, path=metadata_path)
    assert metadata_out == metadata_in

    with tables.open_file(filename, "r") as file:
        metadata_out = meta.read_hdf5_metadata(file, path=metadata_path)

    assert metadata_out == metadata_in


def test_product_metadata_hdf5(tmp_path, ctao_product):
    filename = tmp_path / "product.h5"

    with tables.open_file(filename, mode="w") as h5file:
        meta.write_product_metadata(ctao_product, h5file)
        h5file.root._v_attrs["unrelated"] = "metadata"

        attributes = meta.read_hdf5_metadata(h5file)
        assert attributes["CTAO.data.level"] == "DL1"
        assert attributes["CTAO.instance.obs_id"] == 42
        assert attributes["CTAO.contact.email"] == "test@example.org"
        assert attributes["CTAO.activity.name"] == "ctapipe-process"
        assert attributes["CTAO.model.version"] == "1.0"

        product = meta.read_product_metadata(h5file)

    assert product == ctao_product
    assert attributes["unrelated"] == "metadata"


def test_product_metadata_hdf5_non_root(tmp_path, ctao_product):
    filename = tmp_path / "product.h5"
    metadata_path = "/node/subnode"

    with tables.open_file(filename, mode="w") as h5file:
        h5file.create_group(where="/node", name="subnode", createparents=True)
        meta.write_product_metadata(ctao_product, h5file, path=metadata_path)

    product = meta.read_product_metadata(filename, path=metadata_path)
    assert product == ctao_product


def test_legacy_reference_to_product(reference, ctao_product_type):
    product = meta._legacy_reference_to_product(reference, ctao_product_type)

    assert isinstance(product, dp.Product)
    assert product.description == reference.product.description
    assert product.creation_time == reference.product.creation_time
    assert product.data == ctao_product_type
    assert product.instance.id == uuid.UUID(reference.product.id_)
    assert product.instance.category is None

    assert product.contact.name == reference.contact.name
    assert product.contact.organization == reference.contact.organization
    assert product.contact.email == reference.contact.email

    assert product.model.name == reference.product.data_model_name
    assert product.model.version == reference.product.data_model_version
    assert str(product.model.url) == reference.product.data_model_url + "/"

    assert product.activity.name == reference.activity.name
    assert product.activity.id == uuid.UUID(reference.activity.id_)
    assert product.activity.start == reference.activity.start_time
    assert product.activity.end == reference.activity.stop_time
    assert product.activity.software.name == reference.activity.software_name
    assert product.activity.software.version == reference.activity.software_version
    assert product.activity.software.url is None


@pytest.mark.parametrize(
    ("legacy_levels", "current_level"),
    [
        ([DataLevel.R0], dp.DataLevel.R0),
        (
            [DataLevel.DL1_IMAGES, DataLevel.DL1_PARAMETERS, DataLevel.DL1_MUON],
            dp.DataLevel.DL1,
        ),
        ([DataLevel.R1, DataLevel.DL2], dp.DataLevel.DL2),
    ],
)
def test_legacy_reference_data_level(reference, legacy_levels, current_level):
    reference.product.data_levels = legacy_levels
    product_type = dp.ProductType(
        level=current_level,
        division="Event",
        association="Subarray",
        type="ObservationSim",
    )

    product = meta._legacy_reference_to_product(reference, product_type)
    assert product.data.level is current_level


@pytest.mark.parametrize("category", ["A", "B", "C"])
def test_legacy_reference_processing_category(reference, ctao_product_type, category):
    reference.product.data_category = category

    product = meta._legacy_reference_to_product(reference, ctao_product_type)
    assert product.instance.category is dp.DataProcessingCategory(category)


def test_legacy_reference_ignores_fields_without_equivalent(
    reference, ctao_product_type
):
    reference.product.format = "legacy-format"
    reference.process.type_ = "Simulation"
    reference.process.subtype = "legacy-process-subtype"
    reference.instrument.type_ = "legacy-instrument-type"
    reference.instrument.subtype = "legacy-instrument-subtype"
    reference.instrument.version = "legacy-instrument-version"

    product = meta._legacy_reference_to_product(reference, ctao_product_type)
    serialized = product.model_dump_json()

    assert "legacy-format" not in serialized
    assert "legacy-process-subtype" not in serialized
    assert "legacy-instrument-type" not in serialized
    assert "legacy-instrument-subtype" not in serialized
    assert "legacy-instrument-version" not in serialized
    assert product.activity.process is dp.ObservatoryProcess.USER
    assert product.instance.site_id is None


def test_legacy_reference_ignores_invalid_product_id(reference, ctao_product_type):
    reference.product.id_ = "not-a-uuid"

    product = meta._legacy_reference_to_product(reference, ctao_product_type)
    assert isinstance(product.instance.id, uuid.UUID)


def test_legacy_reference_checks_product_type(reference, ctao_product_type):
    wrong_association = ctao_product_type.model_copy(
        update={"association": dp.DataAssociation.TELESCOPE}
    )
    with pytest.raises(ValueError, match="association"):
        meta._legacy_reference_to_product(reference, wrong_association)

    wrong_level = ctao_product_type.model_copy(update={"level": dp.DataLevel.DL2})
    with pytest.raises(ValueError, match="levels"):
        meta._legacy_reference_to_product(reference, wrong_level)


def test_reprs(reference):
    assert isinstance(repr(reference), str)
    assert isinstance(repr(reference.activity), str)
    assert isinstance(repr(reference.product), str)
    assert isinstance(repr(reference.contact), str)
    assert isinstance(repr(reference.instrument), str)
    assert isinstance(repr(reference.process), str)


def test_read_current_metadata(reference_file):
    product = meta.read_ctao_metadata(reference_file)

    assert isinstance(product, dp.Product)
    assert product.description == "ctapipe test product"


def test_read_legacy_metadata(legacy_reference_file):
    product = meta.read_ctao_metadata(legacy_reference_file)

    assert isinstance(product, dp.Product)
    assert product.description == "An Amazing Product"


def test_read_legacy_metadata_requires_product_type(legacy_reference_file):
    with pytest.raises(ValueError, match="product_type"):
        meta.read_ctao_metadata(legacy_reference_file)


def test_read_unknown_metadata(tmp_path):
    filename = tmp_path / "unknown_metadata.h5"

    with tables.open_file(filename, mode="w") as h5file:
        h5file.root._v_attrs["SOME METADATA"] = "unknown"

    with pytest.raises(ValueError, match="Unsupported metadata format"):
        meta.read_ctao_metadata(filename)
