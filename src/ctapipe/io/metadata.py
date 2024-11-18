"""
Management of CTAO Reference Metadata.

Definitions from :cite:`ctao-top-level-data-model`.
This information is required to be attached to the header of any files generated.

The class Reference collects all required reference metadata, and can be turned into a
flat dictionary. The user should try to fill out all fields, or use a helper to fill
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
import gzip
import os
import uuid
import warnings
from collections import defaultdict
from contextlib import ExitStack

import tables
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from tables import NaturalNameWarning
from traitlets import Enum, HasTraits, Instance, List, Unicode, UseEnum, default
from traitlets.config import Configurable

from ..core.traits import AstroTime
from .datalevels import DataLevel

__all__ = [
    "Reference",
    "Contact",
    "Process",
    "Product",
    "Activity",
    "Instrument",
    "write_to_hdf5",
    "read_hdf5_metadata",
    "read_reference_metadata",
]


CONVERSIONS = {
    Time: lambda value: value.utc.iso,
    list: lambda value: ",".join([convert(elem) for elem in value]),
    DataLevel: lambda value: value.name,
}


def convert(value):
    """Convert to representation suitable for header infos, such as hdf5 or fits"""
    if (conv := CONVERSIONS.get(type(value))) is not None:
        return conv(value)
    return value


def _get_user_name():
    """return the logged in user's name, as a fall-back if none is specified"""
    try:
        import pwd

        return pwd.getpwuid(os.getuid()).pw_gecos
    except ImportError:
        # the pwd module is not available on some non-unix systems (Windows), so
        # here we just fall back to a default name
        return "Unknown User"


class Contact(Configurable):
    """Contact information"""

    name = Unicode("unknown").tag(config=True)
    email = Unicode("unknown").tag(config=True)
    organization = Unicode("unknown").tag(config=True)

    @default("name")
    def default_name(self):
        """if no name specified, use the system's user name"""
        try:
            return _get_user_name()
        except RuntimeError:
            return ""

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', email='{self.email}'"
            f", organization='{self.organization}'"
            ")"
        )


class Product(HasTraits):
    """Data product information"""

    description = Unicode("unknown")
    creation_time = AstroTime()
    id_ = Unicode(help="leave unspecified to automatically generate a UUID")
    data_category = Enum(["Sim", "A", "B", "C", "Other"], "Other")
    data_levels = List(UseEnum(DataLevel))
    data_association = Enum(["Subarray", "Telescope", "Target", "Other"], "Other")
    data_model_name = Unicode("unknown")
    data_model_version = Unicode("unknown")
    data_model_url = Unicode("unknown")
    format = Unicode()

    def __init__(self, **kwargs):
        if "data_levels" in kwargs:
            data_levels = kwargs["data_levels"]
            if isinstance(data_levels, str):
                if data_levels.strip() == "":
                    kwargs["data_levels"] = []
                else:
                    kwargs["data_levels"] = data_levels.split(",")

        super().__init__(**kwargs)

    @default("creation_time")
    def default_time(self):
        """return current time by default"""
        return Time.now().iso

    @default("id_")
    def default_product_id(self):
        """default id is a UUID"""
        return str(uuid.uuid4())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"id_='{self.id_}'"
            f", description='{self.description}'"
            f", creation_time='{self.creation_time.utc.isot}Z'"
            f", data_category='{self.data_category}'"
            f", data_levels='{','.join(dl.name for dl in self.data_levels)}'"
            f", data_model_version='{self.data_model_version}'"
            f", format='{self.format}'"
            ")"
        )


class Process(HasTraits):
    """Process (top-level workflow) information"""

    type_ = Enum(["Observation", "Simulation", "Other"], "Other")
    subtype = Unicode("")
    id_ = Unicode("")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"id_='{self.id_}'"
            f", type_='{self.type_}'"
            f", subtype='{self.subtype}'"
            ")"
        )


class Activity(HasTraits):
    """Activity (tool) information"""

    @classmethod
    def from_provenance(cls, activity):
        """construct Activity metadata from existing ActivityProvenance object"""
        return Activity(
            name=activity["activity_name"],
            type_="software",
            id_=activity["activity_uuid"],
            start_time=activity["start"]["time_utc"],
            stop_time=activity["stop"].get("time_utc", Time.now()),
            software_name="ctapipe",
            software_version=activity["system"]["ctapipe_version"],
        )

    name = Unicode()
    type_ = Unicode("software")
    id_ = Unicode()
    start_time = AstroTime()
    stop_time = AstroTime(allow_none=True, default_value=None)
    software_name = Unicode("unknown")
    software_version = Unicode("unknown")

    # pylint: disable=no-self-use
    @default("start_time")
    def default_time(self):
        """default time is now"""
        return Time.now().iso

    def __repr__(self):
        if self.stop_time is not None:
            stop_time = f"'{self.stop_time.utc.isot}Z'"
        else:
            stop_time = None
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}'"
            f", id_='{self.id_}'"
            f", type_='{self.type_}'"
            f", start_time='{self.start_time.utc.isot}Z'"
            f", stop_time={stop_time}"
            f", software_name='{self.software_name}'"
            f", software_version='{self.software_version}'"
            ")"
        )


class Instrument(Configurable):
    """Instrumental Context"""

    site = Unicode(
        default_value="Other",
        help=(
            "Which site of CTAO (or external telescope)"
            " this instrument is associated with"
        ),
    ).tag(config=True)

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
    ).tag(config=True)

    type_ = Unicode("unspecified").tag(config=True)
    subtype = Unicode("unspecified").tag(config=True)
    version = Unicode("unspecified").tag(config=True)
    id_ = Unicode("unspecified").tag(config=True)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"site='{self.site}', class_='{self.class_}', type_='{self.type_}'"
            f", subtype='{self.subtype}', version='{self.version}', id_='{self.id_}'"
            ")"
        )


def _to_dict(hastraits_instance, prefix=""):
    """helper to convert a HasTraits to a dict with keys
    in the required CTAO format (upper-case, space separated)
    """
    res = {}

    ignore = {"parent", "config"}
    for k, trait in hastraits_instance.traits().items():
        if k in ignore:
            continue

        key = (prefix + k.upper().replace("_", " ")).replace("  ", " ").strip()
        val = trait.get(hastraits_instance)

        # apply type conversions
        val = convert(val)
        res[key] = val

    return res


class Reference(HasTraits):
    """All the reference Metadata required for a CTAO output file, plus a way to turn
    it into a dict() for easy addition to the header of a file"""

    contact = Instance(Contact)
    product = Instance(Product)
    process = Instance(Process)
    activity = Instance(Activity)
    instrument = Instance(Instrument)

    def to_dict(self, fits=False):
        """
        convert Reference metadata to a flat dict.

        If ``fits=True``, this will include the ``HIERARCH`` keyword in front.
        """
        prefix = "CTA " if fits is False else "HIERARCH CTA "

        meta = {prefix + "REFERENCE VERSION": "1"}
        meta.update(_to_dict(self.contact, prefix=prefix + "CONTACT "))
        meta.update(_to_dict(self.product, prefix=prefix + "PRODUCT "))
        meta.update(_to_dict(self.process, prefix=prefix + "PROCESS "))
        meta.update(_to_dict(self.activity, prefix=prefix + "ACTIVITY "))
        meta.update(_to_dict(self.instrument, prefix=prefix + "INSTRUMENT "))
        return meta

    @classmethod
    def from_dict(cls, metadata):
        kwargs = defaultdict(dict)
        for hierarchical_key, value in metadata.items():
            components = hierarchical_key.split(" ")

            if components[0] != "CTA":
                continue

            if len(components) < 3:
                continue

            group = components[1].lower()
            key = "_".join(components[2:]).lower()

            # handle python builtins / keywords
            if key in {"type", "id", "class"}:
                key = key + "_"

            kwargs[group][key] = value

        return cls(
            contact=Contact(**kwargs["contact"]),
            product=Product(**kwargs["product"]),
            process=Process(**kwargs["process"]),
            activity=Activity(**kwargs["activity"]),
            instrument=Instrument(**kwargs["instrument"]),
        )

    @classmethod
    def from_fits(cls, header):
        # for now, just use from_dict, but we might need special handling
        # of some keys
        return cls.from_dict(header)

    def __repr__(self):
        return str(self.to_dict())


def write_to_hdf5(metadata, h5file, path="/"):
    """
    Write metadata fields to a PyTables HDF5 file handle.

    Parameters
    ----------
    metadata: dict
        flat dict as generated by `Reference.to_dict()`
    h5file: string, Path, or `tables.file.File`
        pytables filehandle
    path: string
        default: '/' is the path to ctapipe global metadata
        the node must already exist in the 5hfile
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NaturalNameWarning)
        node = h5file.get_node(path)
        for key, value in metadata.items():
            node._v_attrs[key] = value  # pylint: disable=protected-access


def read_hdf5_metadata(h5file, path="/"):
    """
    Read hdf5 attributes into a dict
    """
    with ExitStack() as stack:
        if not isinstance(h5file, tables.File):
            h5file = stack.enter_context(tables.open_file(h5file))

        node = h5file.get_node(path)
        return {key: node._v_attrs[key] for key in node._v_attrs._f_list()}


def read_reference_metadata(path):
    """Read CTAO data product metadata from path

    File is first opened to determine file format, then the metadata
    is read. Supported are currently FITS and HDF5.
    """
    header_bytes = 8
    with open(path, "rb") as f:
        first_bytes = f.read(header_bytes)

    if first_bytes.startswith(b"\x1f\x8b"):
        with gzip.open(path, "rb") as f:
            first_bytes = f.read(header_bytes)

    if first_bytes.startswith(b"\x89HDF"):
        return _read_reference_metadata_hdf5(path)

    if first_bytes.startswith(b"SIMPLE"):
        return _read_reference_metadata_fits(path)

    if first_bytes.startswith(b"# %ECSV"):
        return Reference.from_dict(Table.read(path).meta)

    raise ValueError(
        f"'{path}' is not one of the supported file formats: fits, hdf5, ecsv"
    )


def _read_reference_metadata_hdf5(h5file, path="/"):
    meta = read_hdf5_metadata(h5file, path)
    return Reference.from_dict(meta)


def _read_reference_metadata_ecsv(path):
    return Reference.from_dict(Table.read(path).meta)


def _read_reference_metadata_fits(fitsfile, hdu: int | str = 0):
    """
    Read reference metadata from a fits file

    Parameters
    ----------
    fitsfile: string, Path, or `tables.file.File`
        hdf5 file
    hdu: int or str
        HDU index or name.

    Returns
    -------
    reference_metadata: Reference
    """
    with ExitStack() as stack:
        if not isinstance(fitsfile, fits.HDUList):
            fitsfile = stack.enter_context(fits.open(fitsfile))

        return Reference.from_fits(fitsfile[hdu].header)
