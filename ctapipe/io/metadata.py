"""
Management of CTA Reference Metadata, as defined in the CTA Top-Level Data Model
document, version 1A. This information is required to be attached to the header of any
files generated.

The class Reference collects all required reference metadata, and can be turned into a
flat dictionary.  The user should try to fill out all fields, or use a helper to fill
them (as in `Activity.from_provenance()`)

.. code-block:: python

    ref = Reference(
        contact=Contact(name="Some User", email="user@me.com"),
        product=Product(format='hdf5', ...),
        process=Process(...),
        activity=Activity(...),
        instrument = Instrument(...)
    )

    some_astropy_table.meta = ref.to_dict()
    some_astropy_table.write("output.ecsv")

"""
import uuid
from collections import OrderedDict

from astropy.time import Time
from traitlets import (
    Enum,
    Unicode,
    Int,
    HasTraits,
    default,
    Instance,
    validate,
)

__all__ = ["Reference", "Contact", "Process", "Product", "Activity", "Instrument"]


class Contact(HasTraits):
    """ Contact information """

    name = Unicode("unknown")
    email = Unicode("unknown")
    organization = Unicode("unknown")


class Product(HasTraits):
    """Data product information"""

    description = Unicode("unknown")
    creation_time = Unicode()
    _id = Unicode(help="leave unspecified to automatically generate a UUID")
    data_category = Enum(["S", "A", "B", "C", "Other"], "Other")
    data_level = Enum(
        ["R0", "R1", "DL0", "DL1", "DL2", "DL3", "DL4", "DL5", "DL6", "Other"], "Other"
    )
    data_association = Enum(["Subarray", "Telescope", "Target", "Other"], "Other")
    data_model_name = Unicode("unknown")
    data_model_version = Unicode("unknown")
    data_model_url = Unicode("unknown")
    format = Unicode()

    @default("creation_time")
    def default_time(self):
        return Time.now().iso

    @validate("creation_time")
    def valid_time(selfs, proposal):
        thetime = Time(proposal["value"])
        return thetime.iso

    @default("_id")
    def default_product_id(self):
        return str(uuid.uuid4())


class Process(HasTraits):
    """ Process (top-level workflow) information """

    _type = Enum(["Observation", "Simulation", "Other"], "Other")
    subtype = Unicode("")
    _id = Int()


class Activity(HasTraits):
    """ Activity (tool) information """

    @classmethod
    def from_provenance(cls, activity: "ActivityProvenance"):
        return Activity(
            name=activity["activity_name"],
            _type="software",
            _id=activity["activity_uuid"],
            start=activity["start"]["time_utc"],
            software_name="ctapipe",
            software_version=activity["system"]["ctapipe_version"],
        )

    name = Unicode()
    _type = Unicode("software")
    _id = Unicode()
    start_time = Unicode()
    software_name = Unicode("unknown")
    software_version = Unicode("unknown")

    @default("start_time")
    def default_time(self):
        return Time.now().iso


class Instrument(HasTraits):
    """ Instrumental Context """

    site = Enum(["CTA-North", "CTA-South", "Other"], "Other")
    _class = Enum(
        [
            "array",
            "subarray",
            "telescope",
            "camera",
            "optics",
            "mirror",
            "photosensor",
            "module",
            "part",
            "other",
        ],
        "other",
    )
    _type = Unicode("unspecified")
    subtype = Unicode("unspecified")
    version = Unicode("unspecified")
    _id = Unicode("unspecified")


def _to_dict(x, prefix=""):
    """ helper to convert a HasTraits to a dict with keys
    in the required CTA format (upper-case, space separated)
    """
    return {
        (prefix + k.upper().replace("_", " ")).replace("  ", " "): tr.get(x)
        for k, tr in x.traits().items()
    }


class Reference(HasTraits):
    """ All the reference Metadata required for a CTA output file, plus a way to turn
    it into a dict() for easy addition to the header of a file """

    contact = Instance(Contact)
    product = Instance(Product)
    process = Instance(Process)
    activity = Instance(Activity)
    instrument = Instance(Instrument)

    def to_dict(self):
        meta = OrderedDict({"CTA REFERENCE VERSION": "1"})
        meta.update(_to_dict(self.contact, prefix="CTA CONTACT "))
        meta.update(_to_dict(self.product, prefix="CTA PRODUCT "))
        meta.update(_to_dict(self.process, prefix="CTA PROCESS "))
        meta.update(_to_dict(self.activity, prefix="CTA ACTIVITY "))
        meta.update(_to_dict(self.instrument, prefix="CTA INSTRUMENT "))
        return meta
