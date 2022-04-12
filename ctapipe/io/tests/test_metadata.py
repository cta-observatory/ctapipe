"""
Test CTA Reference metadata functionality
"""

import tables
from ctapipe.io import metadata as meta
from ctapipe.core.provenance import Provenance


def test_construct_and_write_metadata(tmp_path):
    """ basic test of making a Reference object and writing it"""

    prov = Provenance()
    prov.start_activity("test")
    prov.finish_activity()
    prov_activity = prov.finished_activities[0]

    reference = meta.Reference(
        contact=meta.Contact(
            name="Somebody", email="a@b.com", organization="CTA Consortium"
        ),
        product=meta.Product(
            description="An Amazing Product",
            creation_time="2020-10-11 15:23:31",
            data_category="Sim",
            data_level=["DL1_IMAGES", "DL1_PARAMETERS"],
            data_association="Subarray",
            data_model_name="Unofficial DL1",
            data_model_version="1.0",
            data_model_url="http://google.com",
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

    ref_dict = reference.to_dict()
    assert ref_dict["CTA PRODUCT FORMAT"] == "hdf5"

    import uuid  # pylint: disable=import-outside-toplevel

    assert str(uuid.UUID(ref_dict["CTA PRODUCT ID"])) == ref_dict["CTA PRODUCT ID"]

    # check that we can write this to the header of a typical table file in multiple
    # formats:
    from astropy.table import Table  # pylint: disable=import-outside-toplevel

    table = Table(dict(x=[1, 2, 3], y=[15.2, 15.2, 14.5]))
    for path in [tmp_path / "test.fits", tmp_path / "test.ecsv"]:
        if ".fits" in path.suffixes:
            reference.format = "fits"
            ref_dict = reference.to_dict(fits=True)
        else:
            reference.format = "ecsv"
            ref_dict = reference.to_dict()

        table.meta = ref_dict
        table.write(path)

    # write to pytables file

    import tables  # pylint: disable=import-outside-toplevel

    with tables.open_file(tmp_path / "test.h5", mode="w") as h5file:
        h5file.create_group(where='/', name='node')
        meta.write_to_hdf5(ref_dict, h5file, path='/node')


def test_read_metadata(tmp_path):
    # Testing one can read both a path as well as a PyTables file object
    filename = tmp_path / "test.h5"
    metadata_in = {
        'SOFTWARE': 'ctapipe',
        'FOO': 'BAR'
    }
    metadata_path = '/node/subnode'
    with tables.open_file(filename, mode="w") as h5file:
        h5file.create_group(where='/node', name='subnode', createparents=True)
        meta.write_to_hdf5(metadata_in, h5file, path=metadata_path)

    metadata_out = meta.read_metadata(filename, path=metadata_path)
    assert metadata_out == metadata_in

    with tables.open_file(filename, 'r') as file:
        metadata_out = meta.read_metadata(file, path=metadata_path)

    assert metadata_out == metadata_in
