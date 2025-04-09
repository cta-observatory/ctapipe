"""
Test CTA Reference metadata functionality
"""
import uuid

import pytest
import tables
from astropy.io import fits
from astropy.table import Table

from ctapipe.core.provenance import Provenance
from ctapipe.io import metadata as meta


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


def test_reprs(reference):
    assert isinstance(repr(reference), str)
    assert isinstance(repr(reference.activity), str)
    assert isinstance(repr(reference.product), str)
    assert isinstance(repr(reference.contact), str)
    assert isinstance(repr(reference.instrument), str)
    assert isinstance(repr(reference.process), str)
