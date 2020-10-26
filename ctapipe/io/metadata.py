"""
Management of CTA Reference Metadata, as defined in the CTA Top-Level Data Model
document [ctatopleveldatamodel]_ , version 1A. This information is required to be
attached to the header of any
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
import warnings
from collections import OrderedDict

from traitlets import Enum, Unicode, Int, HasTraits, default, Instance

from ctapipe.core.provenance import _ActivityProvenance
from ctapipe.core.traits import AstroTime

__all__ = [
    "Reference",
    "Contact",
    "Process",
    "Product",
    "Activity",
    "Instrument",
    "write_to_hdf5",
]

from astropy.time import Time


CONVERSIONS = {Time: lambda t: t.utc.iso}


class Contact(HasTraits):
    """ Contact information """

    name = Unicode("unknown")
    email = Unicode("unknown")
    organization = Unicode("unknown")


class Product(HasTraits):
    """Data product information"""

    description = Unicode("unknown")
    creation_time = AstroTime()
    id_ = Unicode(help="leave unspecified to automatically generate a UUID")
    data_category = Enum(["S", "A", "B", "C", "Other"], "Other")
    data_level = Enum(
        ["R0", "R1", "DL0", "DL1", "DL2", "DL3", "DL4", "DL5", "DL6", "Other"], "Other"
    )
    data_association = Enum(["Subarray", "Telescope", "Target", "Other"], "Other")
    data_model_name = Unicode("unknown")
    data_model_version = Unicode("unknown")
    data_model_url = Unicode("unknown")
    format = Unicode()

    # pylint: disable=no-self-use
    @default("creation_time")
    def default_time(self):
        """ return current time by default """
        return Time.now().iso

    @default("id_")
    def default_product_id(self):
        """ default id is a UUID """
        return str(uuid.uuid4())


class Process(HasTraits):
    """ Process (top-level workflow) information """

    type_ = Enum(["Observation", "Simulation", "Other"], "Other")
    subtype = Unicode("")
    id_ = Int()


class Activity(HasTraits):
    """ Activity (tool) information """

    @classmethod
    def from_provenance(cls, activity: _ActivityProvenance):
        """ construct Activity metadata from existing ActivityProvenance object"""
        return Activity(
            name=activity["activity_name"],
            type_="software",
            id_=activity["activity_uuid"],
            start_time=activity["start"]["time_utc"],
            software_name="ctapipe",
            software_version=activity["system"]["ctapipe_version"],
        )

    name = Unicode()
    type_ = Unicode("software")
    id_ = Unicode()
    start_time = AstroTime()
    software_name = Unicode("unknown")
    software_version = Unicode("unknown")

    # pylint: disable=no-self-use
    @default("start_time")
    def default_time(self):
        """ default time is now """
        return Time.now().iso


class Instrument(HasTraits):
    """ Instrumental Context """

    site = Enum(
        [
            "CTA-North",
            "CTA-South",
            "SDMC",
            "HQ",
            "MAGIC",
            "HESS",
            "VERITAS",
            "FACT",
            "Whipple",
            "Other",
        ],
        "Other",
        help="Which site of CTA (or external telescope) "
        "this instrument is associated with",
    )
    class_ = Enum(
        [
            "Array",
            "Subarray",
            "Telescope",
            "Camera",
            "Optics",
            "Mirror",
            "Photo-sensor",
            "Module",
            "Part",
            "Other",
        ],
        "Other",
    )
    type_ = Unicode("unspecified")
    subtype = Unicode("unspecified")
    version = Unicode("unspecified")
    id_ = Unicode("unspecified")


def _to_dict(hastraits_instance, prefix=""):
    """ helper to convert a HasTraits to a dict with keys
    in the required CTA format (upper-case, space separated)
    """
    res = {}

    for k, tr in hastraits_instance.traits().items():
        key = (prefix + k.upper().replace("_", " ")).replace("  ", " ").strip()
        val = tr.get(hastraits_instance)

        # apply type conversions
        val = CONVERSIONS.get(type(val), lambda v: v)(val)
        res[key] = val

    return res


class Reference(HasTraits):
    """ All the reference Metadata required for a CTA output file, plus a way to turn
    it into a dict() for easy addition to the header of a file """

    contact = Instance(Contact)
    product = Instance(Product)
    process = Instance(Process)
    activity = Instance(Activity)
    instrument = Instance(Instrument)

    def to_dict(self, fits=False):
        """
        convert Reference metadata to a flat dict.

        If `fits=True`, this will include the `HIERARCH` keyword in front.
        """
        prefix = "CTA " if fits is False else "HIERARCH CTA "

        meta = OrderedDict({prefix + "REFERENCE VERSION": "1"})
        meta.update(_to_dict(self.contact, prefix=prefix + "CONTACT "))
        meta.update(_to_dict(self.product, prefix=prefix + "PRODUCT "))
        meta.update(_to_dict(self.process, prefix=prefix + "PROCESS "))
        meta.update(_to_dict(self.activity, prefix=prefix + "ACTIVITY "))
        meta.update(_to_dict(self.instrument, prefix=prefix + "INSTRUMENT "))
        return meta


def write_to_hdf5(metadata, h5file):
    """
    Write metadata fields to a PyTables HDF5 file handle.

    Parameters
    ----------
    metadata: dict
        flat dict as generated by `Reference.to_dict()`
    h5file: tables.file.File
        pytables filehandle
    """
    from tables import NaturalNameWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NaturalNameWarning)
        for key, value in metadata.items():
            h5file.root._v_attrs[key] = value  # pylint: disable=protected-access
