"""Read, write, and migrate CTAO data-product metadata.

This module provides serialization helpers for current CTAO product metadata defined
by :mod:`ctao_datamodel`, including reading metadata from HDF5, FITS, ECSV, and JSON
files and writing flattened product metadata to HDF5 attributes.

Legacy CTA reference metadata remains supported through :class:`Reference` and its
component classes. :func:`read_ctao_metadata` transparently converts such metadata to
the current :class:`ctao_datamodel.models.dataproducts.Product` model, while
:func:`read_reference_metadata` provides access to the original legacy representation.
"""

import gzip
import os
import uuid
import warnings
from collections import defaultdict
from collections.abc import Iterable
from contextlib import ExitStack

import ctao_datamodel as dm
import ctao_datamodel.models.dataproducts as dp
import tables
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from ctao_datamodel.models.common import SiteID
from pydantic import ValidationError
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
    "convert",
    "get_compatible_metadata_versions",
    "write_to_hdf5",
    "write_product_metadata",
    "read_reference_metadata",
    "read_ctao_metadata",
    "to_ctao_data_level",
    "to_ctao_data_type",
    "to_ctao_data_association",
]


CONVERSIONS = {
    Time: lambda value: value.utc.iso,
    list: lambda value: ",".join([convert(elem) for elem in value]),
    DataLevel: lambda value: value.name,
}


def convert(value):
    """Convert a metadata value to a representation suitable for file headers.

    Values with a registered conversion are serialized to scalar or string values
    supported by formats such as HDF5 and FITS. Other values are returned unchanged.

    Parameters
    ----------
    value
        Metadata value to convert.

    Returns
    -------
    object
        The converted value, or the original value if no conversion is registered.
    """
    if (conv := CONVERSIONS.get(type(value))) is not None:
        return conv(value)
    return value


def _get_user_name():
    """return the logged in user's name, as a fall-back if none is specified"""
    try:
        import pwd

        return pwd.getpwuid(os.getuid()).pw_gecos
    except Exception:
        # the pwd module is not available on some non-unix systems (Windows)
        # also, a username might not exist (e.g. in docker containers run with a custom uid)
        # so here we just fall back to a default name
        return "Unknown User"


class Contact(Configurable):
    """Legacy CTA reference-metadata contact information.

    This configurable class represents the person or organization responsible for a
    data product written with the legacy CTA metadata schema.
    """

    name = Unicode("unknown").tag(config=True)
    email = Unicode("unknown@example.org").tag(config=True)
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
    """Legacy CTA reference-metadata description of a data product.

    The fields describe the product identity, data levels, processing category,
    association, data model, and storage format used by the legacy metadata schema.
    """

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
    """Legacy CTA reference metadata for the top-level producing process."""

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
    """Legacy CTA reference metadata for the activity producing a data product."""

    @classmethod
    def from_provenance(cls, activity):
        """Create legacy activity metadata from a provenance record.

        Parameters
        ----------
        activity : dict
            Serialized ctapipe activity provenance.

        Returns
        -------
        Activity
            Activity metadata populated from the provenance record.
        """
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
    """Legacy CTA reference metadata describing the instrumental context."""

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
    """Complete metadata record using the legacy CTA reference schema.

    A reference combines contact, product, process, activity, and instrument
    metadata. Use :meth:`to_dict` to flatten it into file-header attributes.
    """

    contact = Instance(Contact)
    product = Instance(Product)
    process = Instance(Process)
    activity = Instance(Activity)
    instrument = Instance(Instrument)

    def to_dict(self, fits=False):
        """Convert the reference metadata to a flat dictionary.

        Parameters
        ----------
        fits : bool
            If true, prefix keys with ``HIERARCH`` for use in FITS headers.

        Returns
        -------
        dict
            Flattened legacy metadata with CTA header keywords.
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
        """Create a legacy reference record from flattened CTA metadata.

        Parameters
        ----------
        metadata : collections.abc.Mapping
            Metadata containing flattened ``CTA ...`` keys. Unrelated keys are
            ignored.

        Returns
        -------
        Reference
            Parsed legacy reference metadata.
        """
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
        """Create a legacy reference record from a FITS header."""
        # for now, just use from_dict, but we might need special handling
        # of some keys
        return cls.from_dict(header)

    @classmethod
    def from_json(cls, json_data):
        """Create a legacy reference record from a JSON metadata mapping."""
        return cls.from_dict(json_data)

    def __repr__(self):
        return str(self.to_dict())


def read_reference_metadata(path):
    """Read legacy CTA reference metadata from a supported file.

    The format is detected from the file contents. FITS (including gzip-compressed
    FITS), HDF5, ECSV, and JSON are supported.

    Parameters
    ----------
    path : path-like
        File containing legacy CTA reference metadata.

    Returns
    -------
    Reference
        Parsed legacy reference metadata.

    Raises
    ------
    ValueError
        If the file format is not supported.
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

    if first_bytes.startswith(b"{"):
        return _read_reference_metadata_json(path)

    raise ValueError(
        f"'{path}' is not one of the supported file formats: fits, hdf5, ecsv, json"
    )


def _read_reference_metadata_json(path):
    """Read legacy CTA reference metadata from a JSON file."""
    import json

    with open(path) as f:
        data = json.load(f)
    return Reference.from_dict(data.get("metadata", data))


def _read_reference_metadata_hdf5(h5file, path="/"):
    """Read legacy CTA reference metadata from an HDF5 node."""
    meta = _read_hdf5_metadata(h5file, path)
    return Reference.from_dict(meta)


def _read_reference_metadata_ecsv(path):
    """Read legacy CTA reference metadata from an ECSV file."""
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


# -----------------------------------------------------
#  New Data Model
# -----------------------------------------------------


class LegacyMetadataWarning(UserWarning):
    """Warning for incomplete or invalid legacy metadata."""


class LegacyContactRequired(ValueError):
    """Raised when legacy metadata does not contain valid contact information."""


def get_compatible_metadata_versions(
    current_version=None,
) -> set[str]:
    if current_version is None:
        current_version = dp.Product.model_fields["ctao_metadata_version"].default

    migrations = dp.Product.migration_history()

    compatible = {current_version}

    changed = True
    while changed:
        changed = False
        for migration in migrations:
            if migration["to"] in compatible and migration["from"] not in compatible:
                compatible.add(migration["from"])
                changed = True

    return compatible


def read_ctao_metadata(
    input_url,
    *,
    product_type: dp.ProductType | None = None,
    contact_fallback: dp.Contact | None = None,
) -> dp.Product:
    """Read current or legacy CTAO product metadata from a supported file.

    The format is detected from the file contents. FITS (including gzip-compressed
    FITS), HDF5, ECSV, and JSON are supported. Legacy CTA reference metadata is
    converted to the current CTAO data model and emits a
    :class:`LegacyMetadataWarning`.

    Parameters
    ----------
    input_url : path-like or tables.File
        Input file or open PyTables file handle.
    product_type : ctao_datamodel.models.dataproducts.ProductType, optional
        Product type to use when converting legacy metadata. If omitted, it is
        derived from the legacy metadata and, for HDF5, the file contents.
    contact_fallback : ctao_datamodel.models.dataproducts.Contact, optional
        Contact used when legacy contact information is invalid. If omitted, an
        ``unknown`` contact is used.

    Returns
    -------
    ctao_datamodel.models.dataproducts.Product
        Validated metadata using the current CTAO product model.

    Raises
    ------
    ValueError
        If the metadata schema or file format is unsupported.
    pydantic.ValidationError
        If current CTAO metadata does not validate against the product model.
    """
    metadata = _read_raw_metadata(input_url)

    # New Data Model
    if "CTAO.ctao_metadata_version" in metadata:
        return _metadata_to_product(metadata)

    # Old Data Model
    if "CTA REFERENCE VERSION" in metadata:
        warnings.warn(
            "Legacy ctapipe metadata detected. "
            "If this file is not already being migrated, use ctapipe-merge to convert it "
            "to the current CTAO metadata format.",
            LegacyMetadataWarning,
            stacklevel=2,
        )

        reference = Reference.from_dict(metadata)
        if contact_fallback is None:
            contact_fallback = dp.Contact(
                name="unknown",
                organization="unknown",
                email="unknown@example.org",
            )
        if product_type is None:
            product_type = _legacy_product_type(input_url, reference)

        return _legacy_reference_to_product(
            reference, product_type, contact_fallback=contact_fallback
        )

    raise ValueError("Unsupported metadata format")


def _read_raw_metadata(input_file) -> dict:
    """Read flattened metadata from a supported file without validating its schema."""
    if isinstance(input_file, tables.File):
        return _read_hdf5_metadata(input_file)

    # otherwise assume input_file / URL and detect format
    header_bytes = 8

    with open(input_file, "rb") as f:
        first_bytes = f.read(header_bytes)

    if first_bytes.startswith(b"\x1f\x8b"):
        with gzip.open(input_file, "rb") as f:
            first_bytes = f.read(header_bytes)

    if first_bytes.startswith(b"\x89HDF"):
        return _read_hdf5_metadata(input_file)

    if first_bytes.startswith(b"SIMPLE"):
        return _read_fits_metadata(input_file)

    if first_bytes.startswith(b"# %ECSV"):
        return dict(Table.read(input_file).meta)

    if first_bytes.startswith(b"{"):
        return _read_json_metadata(input_file)

    raise ValueError(
        f"'{input_file}' is not one of the supported file formats: fits, hdf5, ecsv, json"
    )


def _read_hdf5_metadata(h5file, path="/"):
    """Read hdf5 attributes into a dict"""
    with ExitStack() as stack:
        if not isinstance(h5file, tables.File):
            h5file = stack.enter_context(tables.open_file(h5file))

        node = h5file.get_node(path)
        return {key: node._v_attrs[key] for key in node._v_attrs._f_list()}


def _read_fits_metadata(path):
    """Read primary-header metadata from a FITS file."""
    with fits.open(path) as hdul:
        return dict(hdul[0].header)


def _read_json_metadata(path):
    """Read metadata from a JSON file or its top-level metadata field."""
    import json

    with open(path) as f:
        data = json.load(f)

    return data.get("metadata", data)


def _metadata_to_product(metadata) -> dp.Product:
    """Convert flattened current CTAO metadata into a validated product model."""
    metadata = {
        key: value for key, value in metadata.items() if key.startswith("CTAO.")
    }

    # Temporary workaround for
    # https://gitlab.cta-observatory.org/cta-computing/common/ctao-datamodel/-/work_items/48
    metadata.setdefault("CTAO.model.url", None)
    metadata.setdefault("CTAO.activity.software.url", None)

    return dm.unflatten_model_instance(
        metadata,
        model=dp.Product,
        parent_key="CTAO",
    )


def _legacy_product_type(input_url, reference: Reference) -> dp.ProductType:
    """Derive a current CTAO product type from legacy reference metadata."""
    level = to_ctao_data_level(reference.product.data_levels)
    association = to_ctao_data_association(reference.product.data_association)
    data_type = to_ctao_data_type(reference, input_url)

    return dp.ProductType(
        level=level,
        division=dp.DataDivision.EVENT,
        association=association,
        type=data_type,
    )


def _legacy_reference_to_product(
    reference: Reference,
    product_type: dp.ProductType,
    contact_fallback: dp.Contact,
) -> dp.Product:
    """Convert legacy reference metadata to a current CTAO Product."""
    instance_kwargs = {"id": _legacy_uuid(reference.product.id_, "product")}

    # Legacy data levels -> processing sublevel
    data_levels = set(reference.product.data_levels)

    has_dl1_images = DataLevel.DL1_IMAGES in data_levels
    has_dl1_parameters = DataLevel.DL1_PARAMETERS in data_levels

    if has_dl1_images and not has_dl1_parameters:
        instance_kwargs["sublevel_id"] = dp.ProcessingSublevel.IMAGES
    elif has_dl1_parameters and not has_dl1_images:
        instance_kwargs["sublevel_id"] = dp.ProcessingSublevel.PARAMETERS

    # Legacy processing category
    try:
        category = dp.DataProcessingCategory(reference.product.data_category)
    except ValueError:
        pass
    else:
        instance_kwargs["category"] = category

    # Legacy instrument site
    site_id = _legacy_site_id(reference.instrument.site)
    if site_id is not None:
        instance_kwargs["site_id"] = site_id

    # Legacy instrument class / id
    instrument_id = _legacy_instrument_id(reference.instrument.id_)

    if reference.instrument.class_ == "Telescope":
        instance_kwargs["ae_class"] = dp.ArrayElementClass.TEL

        if instrument_id is not None:
            instance_kwargs["ae_id"] = instrument_id

    elif reference.instrument.class_ == "Subarray":
        if instrument_id is not None:
            instance_kwargs["subarray_id"] = instrument_id

    model_url = _legacy_optional_string(reference.product.data_model_url)
    contact_name = _legacy_optional_string(reference.contact.name)
    contact_organization = _legacy_optional_string(reference.contact.organization)
    contact_email = _legacy_optional_string(reference.contact.email)

    invalid_contact = {
        "name": contact_name,
        "organization": contact_organization,
        "email": contact_email,
    }

    try:
        contact = dp.Contact(**invalid_contact)
    except ValidationError:
        warnings.warn(
            "Legacy metadata contains invalid contact information: "
            f"{invalid_contact!r}. "
            "The contact information is temporarily replaced with the fallback "
            f"{contact_fallback!r}. "
            "Ensure that valid contact information is provided when writing new data, "
            "for example through the DataWriter, or migrate the file explicitly using "
            "the MergeTool.",
            LegacyMetadataWarning,
            stacklevel=2,
        )
        contact = contact_fallback

    return dp.Product(
        description=reference.product.description,
        creation_time=reference.product.creation_time,
        curation=dp.Curation(),
        data=product_type.model_copy(deep=True),
        instance=dp.InstanceIdentifier(**instance_kwargs),
        model=dp.DataModel(
            name=reference.product.data_model_name,
            version=reference.product.data_model_version,
            url=model_url,
        ),
        contact=contact,
        activity=dp.Activity(
            name=reference.activity.name,
            id=_legacy_uuid(reference.activity.id_, "activity"),
            start=reference.activity.start_time,
            end=reference.activity.stop_time,
            software=dp.Software(
                name=reference.activity.software_name,
                version=reference.activity.software_version,
                url=None,
            ),
            configuration_id="",
        ),
    )


def to_ctao_data_level(data_levels: Iterable[DataLevel]) -> dp.DataLevel:
    """Select the primary CTAO data level from ctapipe data levels.

    DL1 sublevels such as images, parameters, and muon data are normalized to
    ``DL1``. If several levels are present, the highest CTAO data level is returned.

    Parameters
    ----------
    data_levels : collections.abc.Iterable of DataLevel
        ctapipe data levels to convert.

    Returns
    -------
    ctao_datamodel.models.dataproducts.DataLevel
        Primary CTAO data level.

    Raises
    ------
    ValueError
        If ``data_levels`` is empty.
    """
    mapping = {
        DataLevel.DL1_IMAGES: dp.DataLevel.DL1,
        DataLevel.DL1_PARAMETERS: dp.DataLevel.DL1,
        DataLevel.DL1_MUON: dp.DataLevel.DL1,
    }

    mapped_levels = [
        mapping[level] if level in mapping else dp.DataLevel[level.name]
        for level in data_levels
    ]

    if not mapped_levels:
        raise ValueError("At least one data level is required")

    level_order = {level: index for index, level in enumerate(dp.DataLevel)}
    return max(mapped_levels, key=level_order.__getitem__)


LEGACY_MISSING_VALUES = {"", "unknown", "unspecified"}


def _legacy_optional_string(value: str | None) -> str | None:
    """Normalize legacy placeholder strings to ``None``."""
    if value is None:
        return None

    value = str(value).strip()

    if value.lower() in LEGACY_MISSING_VALUES:
        return None

    return value


def _legacy_uuid(value: str | None, field: str) -> uuid.UUID:
    """Parse a legacy UUID, generating a new one for missing or invalid values."""
    try:
        return uuid.UUID(value) if value is not None else uuid.uuid4()
    except (AttributeError, TypeError, ValueError):
        replacement = uuid.uuid4()
        warnings.warn(
            f"Legacy metadata contains an invalid {field} id {value!r}; "
            f"using a newly generated UUID {replacement}.",
            LegacyMetadataWarning,
            stacklevel=2,
        )
        return replacement


def to_ctao_data_type(
    reference: Reference,
    input_file,
) -> dp.DataType:
    """Infer the CTAO data type represented by legacy metadata.

    Explicit legacy process and product fields take precedence. As a final fallback,
    an HDF5 input is inspected for simulation configuration data.

    Parameters
    ----------
    reference : Reference
        Legacy reference metadata.
    input_file : path-like, tables.File, or None
        Input used for the HDF5 simulation fallback. ``None`` is accepted when the
        legacy metadata already determines the result.

    Returns
    -------
    ctao_datamodel.models.dataproducts.DataType
        Inferred observation or simulated-observation data type.
    """
    if reference.process.type_ == "Simulation":
        return dp.DataType.OBSERVATION_SIM

    if reference.process.type_ == "Observation":
        return dp.DataType.OBSERVATION

    if reference.product.data_category == "Sim":
        return dp.DataType.OBSERVATION_SIM

    # final HDF5 fallback
    if isinstance(input_file, tables.File):
        if "/configuration/simulation" in input_file:
            return dp.DataType.OBSERVATION_SIM
    else:
        try:
            with tables.open_file(input_file, mode="r") as h5file:
                if "/configuration/simulation" in h5file:
                    return dp.DataType.OBSERVATION_SIM
        except tables.HDF5ExtError:
            pass

    return dp.DataType.OBSERVATION


def to_ctao_data_association(
    association: str,
) -> dp.DataAssociation:
    """Convert a legacy data association to the CTAO data-model enum.

    Parameters
    ----------
    association : str
        Legacy association value, such as ``"Subarray"`` or ``"Telescope"``.

    Returns
    -------
    ctao_datamodel.models.dataproducts.DataAssociation
        Corresponding CTAO data association.

    Raises
    ------
    ValueError
        If the legacy association has no CTAO equivalent.
    """
    try:
        return dp.DataAssociation(association)
    except ValueError as err:
        raise ValueError(
            f"Unsupported legacy data association: {association!r}"
        ) from err


def _legacy_site_id(site: str | None) -> SiteID | None:
    """Convert an unambiguous legacy instrument site to a CTAO SiteID."""
    site = _legacy_optional_string(site)
    if site is None:
        return None

    mapping = {
        "North": SiteID.CTAO_NORTH,
        "South": SiteID.CTAO_SOUTH,
        "CTAO-North": SiteID.CTAO_NORTH,
        "CTAO-South": SiteID.CTAO_SOUTH,
        "SDMC-DPPS": SiteID.SDMC_DPPS,
        "SDMC-SUSS": SiteID.SDMC_SUSS,
        "HQ": SiteID.HQ,
        "EXTERNAL": SiteID.EXTERNAL,
    }

    return mapping.get(site)


def _legacy_instrument_id(value: str | None) -> int | None:
    """Convert a legacy instrument id to an integer if possible."""
    value = _legacy_optional_string(value)
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return None


def _activity_from_provenance(activity) -> dp.Activity:
    """Create CTAO activity metadata from ctapipe provenance."""
    provenance = activity.provenance

    return dp.Activity(
        process=dp.ObservatoryProcess.DATA_PROCESSING,
        name=provenance["activity_name"],
        id=uuid.UUID(provenance["activity_uuid"]),
        start=provenance["start"]["time_utc"],
        end=provenance["stop"].get("time_utc", Time.now()),
        software=dp.Software(
            name="ctapipe",
            version=provenance["system"]["ctapipe_version"],
            url=None,
        ),
        configuration_id="",
    )


def write_product_metadata(
    product: dp.Product, h5file: tables.File, path="/", remove_legacy=False
):
    """Write a current CTAO product as flattened HDF5 attributes.

    Parameters
    ----------
    product : ctao_datamodel.models.dataproducts.Product
        Validated CTAO product metadata to serialize.
    h5file : tables.File
        Open PyTables file handle.
    path : str
        Path of the existing HDF5 node receiving the attributes.
    remove_legacy : bool
        Remove legacy CTA reference attributes from the target node before writing.
    """
    metadata = dm.flatten_model_instance(
        product,
        parent_key="CTAO",
    )
    if remove_legacy:
        _remove_legacy_metadata(h5file, path=path)
    write_to_hdf5(metadata, h5file, path=path)


def _remove_legacy_metadata(h5file, path="/"):
    """Remove legacy ctapipe reference metadata attributes."""
    node = h5file.get_node(path)

    legacy_prefixes = (
        "CTA REFERENCE ",
        "CTA CONTACT ",
        "CTA PRODUCT ",
        "CTA PROCESS ",
        "CTA ACTIVITY ",
        "CTA INSTRUMENT ",
    )

    for name in node._v_attrs._f_list("user"):
        if name.startswith(legacy_prefixes):
            del node._v_attrs[name]


def write_to_hdf5(metadata, h5file, path="/"):
    """Write flattened metadata as attributes of an HDF5 node.

    Parameters
    ----------
    metadata : collections.abc.Mapping
        Flat metadata, for example as generated by
        :func:`ctao_datamodel.flatten_model_instance` or :meth:`Reference.to_dict`.
    h5file : tables.File
        Open PyTables file handle.
    path : str
        Path of the existing HDF5 node receiving the attributes.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NaturalNameWarning)
        node = h5file.get_node(path)
        for key, value in metadata.items():
            node._v_attrs[key] = value  # pylint: disable=protected-access
