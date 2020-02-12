from ctapipe.io import metadata as meta
from ctapipe.core.provenance import Provenance


def test_construct_and_write_metadata(tmp_path):

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
            data_category="S",
            data_level="DL1",
            data_association="Subarray",
            data_model_name="Unofficial DL1",
            data_model_version="1.0",
            data_model_url="http://google.com",
            format="hdf5",
        ),
        process=meta.Process(_type="Simulation", subtype="Prod3b", _id=423442,),
        activity=meta.Activity.from_provenance(prov_activity.provenance),
        instrument=meta.Instrument(
            site="CTA-North",
            _class="array",
            _type="Layout H1B",
            version="1.0",
            _id="threshold",
        ),
    )

    ref_dict = reference.to_dict()
    assert ref_dict["CTA PRODUCT FORMAT"] == "hdf5"

    import uuid
    assert str(uuid.UUID(ref_dict["CTA PRODUCT ID"])) == ref_dict["CTA PRODUCT ID"]

    # check that we can write this to the header of a typical table file in multiple
    # formats:
    from astropy.table import Table

    table = Table(dict(x=[1, 2, 3], y=[15.2, 15.2, 14.5]))
    table.meta = ref_dict
    for file_name in [tmp_path / "test.fits", tmp_path / "test.ecsv"]:
        table.write(file_name)

    # write to pytables file

    import tables
    with tables.open_file(tmp_path / "test.h5", mode="w") as h5file:
        meta.write_to_hdf5(ref_dict, h5file)
