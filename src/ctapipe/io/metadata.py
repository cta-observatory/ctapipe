"""Read, write, and migrate CTAO data-product metadata.

This module provides helpers for configuring, reading, writing, and converting
CTAO data-product metadata defined by :mod:`ctao_datamodel`. Current product
metadata can be read from HDF5, FITS, ECSV, and JSON files and written to HDF5
attributes or FITS headers.

Legacy ctapipe reference metadata remains supported through :class:`Reference`
and its component classes. :func:`read_reference_metadata` provides access to
the legacy representation, while :func:`read_ctao_metadata` converts legacy
metadata from any supported format to the current
:class:`ctao_datamodel.models.dataproducts.Product` model.
"""

import gzip
import uuid
import warnings
from collections import defaultdict
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path

import ctao_datamodel as dm
import ctao_datamodel.models.dataproducts as dp
import tables
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from ctao_datamodel.models.common import SiteID
from pydantic import TypeAdapter, ValidationError
from tables import NaturalNameWarning
from traitlets import (
    Dict,
    Enum,
    HasTraits,
    Instance,
    List,
    TraitError,
    Unicode,
    UseEnum,
    default,
    observe,
    validate,
)
from traitlets.config import Configurable

from ..core.traits import AstroTime
from ..utils.deprecation import CTAPipeDeprecationWarning
from .datalevels import DataLevel

__all__ = [
    "Activity",
    "Contact",
    "Curation",
    "Instrument",
    "LegacyMetadataWarning",
    "Process",
    "Product",
    "ProductMetadata",
    "Reference",
    "activity_from_provenance",
    "convert",
    "get_compatible_metadata_versions",
    "metadata_to_product",
    "read_ctao_metadata",
    "read_reference_metadata",
    "to_ctao_data_level",
    "write_product_metadata_fits_header",
    "write_product_metadata_hdf5",
    "write_to_hdf5",
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


class Contact(Configurable):
    """Configurable CTAO contact information.

    This class is used for current product metadata configuration and remains
    compatible with the legacy CTA reference metadata schema.
    """

    name = Unicode(default_value="unknown").tag(config=True)
    email = Unicode(default_value="unknown@example.org").tag(config=True)
    organization = Unicode(default_value="unknown").tag(config=True)

    _modified = False

    @observe("name", "email", "organization")
    def _mark_modified(self, change):
        """Record that contact configuration was changed."""
        self._modified = True

    @property
    def modified(self):
        """Whether any configurable contact field was explicitly changed."""
        return self._modified

    @validate("name", "email", "organization")
    def _validate_contact(self, proposal):
        """Validate contact information using the CTAO data model."""
        values = {
            "name": self.name,
            "email": self.email,
            "organization": self.organization,
        }
        values[proposal["trait"].name] = proposal["value"]

        try:
            contact = dp.Contact(**values)
        except ValidationError as err:
            raise TraitError(str(err)) from err
        return getattr(contact, proposal["trait"].name)

    def to_model(self) -> dp.Contact:
        """Return the contact information as a validated CTAO model.

        Returns
        -------
        ctao_datamodel.models.dataproducts.Contact
            Validated contact metadata.

        Raises
        ------
        traitlets.TraitError
            If the configured values do not satisfy the CTAO data model.
        """
        try:
            return dp.Contact(
                name=self.name,
                email=self.email,
                organization=self.organization,
            )
        except ValidationError as err:
            raise TraitError(str(err)) from err

    def to_dict(self):
        """Return the validated contact information as a JSON-compatible mapping."""
        return self.to_model().model_dump(mode="json")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', email='{self.email}'"
            f", organization='{self.organization}'"
            ")"
        )


class Curation(Configurable):
    """Configurable curation metadata for a CTAO data product."""

    release = Unicode(
        default_value=dp.Curation.model_fields["release"].default,
        allow_none=True,
    ).tag(config=True)

    reference = Unicode(
        default_value=dp.Curation.model_fields["reference"].default,
        allow_none=True,
    ).tag(config=True)

    license = Unicode(
        dp.Curation.model_fields["license"].default,
    ).tag(config=True)

    license_url = Unicode(
        dp.Curation.model_fields["license_url"].default,
    ).tag(config=True)

    copyright = Unicode(
        default_value=dp.Curation.model_fields["copyright"].default,
        allow_none=True,
    ).tag(config=True)

    rights = UseEnum(
        dp.DataRights,
        default_value=dp.Curation.model_fields["rights"].default,
        allow_none=True,
    ).tag(config=True)

    release_date = AstroTime(
        default_value=dp.Curation.model_fields["release_date"].default,
        allow_none=True,
    ).tag(config=True)

    valid_from = AstroTime(
        default_value=dp.Curation.model_fields["valid_from"].default,
        allow_none=True,
    ).tag(config=True)

    valid_to = AstroTime(
        default_value=dp.Curation.model_fields["valid_to"].default,
        allow_none=True,
    ).tag(config=True)

    _modified = False

    @observe(
        "release",
        "reference",
        "license",
        "license_url",
        "copyright",
        "rights",
        "release_date",
        "valid_from",
        "valid_to",
    )
    def _mark_modified(self, change):
        """Record that curation configuration was changed."""
        self._modified = True

    @property
    def modified(self):
        """Whether any configurable curation field was explicitly changed."""
        return self._modified

    @validate(
        "release",
        "reference",
        "license",
        "license_url",
        "copyright",
        "rights",
        "release_date",
        "valid_from",
        "valid_to",
    )
    def _validate_curation(self, proposal):
        """Validate curation information using the CTAO data model."""
        values = {
            "release": self.release,
            "reference": self.reference,
            "license": self.license,
            "license_url": self.license_url,
            "copyright": self.copyright,
            "rights": self.rights,
            "release_date": self.release_date,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        }
        name = proposal["trait"].name
        values[name] = proposal["value"]

        try:
            curation = dp.Curation(**values)
        except ValidationError as err:
            raise TraitError(str(err)) from err

        value = getattr(curation, name)

        # dp.Curation.reference is a pydantic AnyUrl,
        # while the configurable trait stores a string.
        if name == "reference" and value is not None:
            return str(value)

        return value

    def to_model(self) -> dp.Curation:
        """Return validated CTAO curation metadata.

        Raises
        ------
        traitlets.TraitError
            If the configured values do not satisfy the CTAO data model.
        """
        try:
            return dp.Curation(
                release=self.release,
                reference=self.reference,
                license=self.license,
                license_url=self.license_url,
                copyright=self.copyright,
                rights=self.rights,
                release_date=self.release_date,
                valid_from=self.valid_from,
                valid_to=self.valid_to,
            )
        except ValidationError as err:
            raise TraitError(str(err)) from err

    def to_dict(self):
        """Return the validated curation information as a JSON-compatible mapping."""
        return self.to_model().model_dump(mode="json")


def _validate_model_field(model, name, value):
    """Validate a single field against a pydantic model."""
    if name not in model.model_fields:
        raise TraitError(f"{name!r} is not a valid field of {model.__name__}")

    field = model.model_fields[name]

    try:
        return TypeAdapter(field.rebuild_annotation()).validate_python(value)
    except ValidationError as err:
        raise TraitError(str(err)) from err


def _validate_model_overrides(model, values, allowed_fields):
    """Validate partial overrides against a pydantic model."""
    unknown = set(values) - allowed_fields
    if unknown:
        raise TraitError(
            f"Fields {sorted(unknown)} cannot be configured for {model.__name__}"
        )

    return {
        name: _validate_model_field(model, name, value)
        for name, value in values.items()
    }


def _apply_model_overrides(model, instance, overrides):
    """Apply validated overrides to an existing pydantic model."""
    if not overrides:
        return instance

    values = instance.model_dump(mode="python")
    values.update(overrides)

    try:
        return model.model_validate(values)
    except ValidationError as err:
        raise TraitError(str(err)) from err


class ProductMetadata(Configurable):
    """User-configurable fields of CTAO product metadata.

    ``description`` and ``disclaimer`` override the corresponding product fields.
    ``instance`` accepts the ``category`` override, while ``activity`` accepts the
    ``configuration_id`` override. Other product metadata is derived by the writer.
    """

    description = Unicode(
        "ctapipe Data Product",
    ).tag(config=True)

    disclaimer = Unicode(
        default_value=None,
        allow_none=True,
    ).tag(config=True)

    instance = Dict(
        default_value={},
    ).tag(config=True)

    activity = Dict(
        default_value={},
    ).tag(config=True)

    _modified = False

    @observe("description", "disclaimer", "instance", "activity")
    def _mark_modified(self, change):
        """Record that product configuration was changed."""
        self._modified = True

    @property
    def modified(self):
        """Whether any configurable product field was explicitly changed."""
        return self._modified

    @validate("description", "disclaimer")
    def _validate_product_metadata(self, proposal):
        """Validate a scalar product field against the CTAO data model."""
        return _validate_model_field(
            dp.Product,
            proposal["trait"].name,
            proposal["value"],
        )

    @validate("instance")
    def _validate_instance(self, proposal):
        """Validate configurable instance-identifier overrides."""
        return _validate_model_overrides(
            dp.InstanceIdentifier,
            proposal["value"],
            allowed_fields={"category"},
        )

    @validate("activity")
    def _validate_activity(self, proposal):
        """Validate configurable activity overrides."""
        return _validate_model_overrides(
            dp.Activity,
            proposal["value"],
            allowed_fields={"configuration_id"},
        )

    def validate_meta(self):
        """Explicitly validate all configurable metadata."""
        _validate_model_field(
            dp.Product,
            "description",
            self.description,
        )
        _validate_model_field(
            dp.Product,
            "disclaimer",
            self.disclaimer,
        )

        _validate_model_overrides(
            dp.InstanceIdentifier,
            self.instance,
            allowed_fields={"category"},
        )

        _validate_model_overrides(
            dp.Activity,
            self.activity,
            allowed_fields={"configuration_id"},
        )

    def to_model(self, **kwargs) -> dp.Product:
        """Create a CTAO product and apply user-configured overrides.

        Parameters
        ----------
        **kwargs
            Values required to construct the product. Configured description,
            disclaimer, instance, and activity values take precedence.

        Returns
        -------
        ctao_datamodel.models.dataproducts.Product
            Validated product metadata.

        Raises
        ------
        traitlets.TraitError
            If the complete product does not satisfy the CTAO data model.
        """
        values = dict(kwargs)

        # User configuration overrides values determined by the writer.
        values["description"] = self.description
        values["disclaimer"] = self.disclaimer

        if "instance" in values:
            values["instance"] = _apply_model_overrides(
                dp.InstanceIdentifier,
                values["instance"],
                self.instance,
            )

        if "activity" in values:
            values["activity"] = _apply_model_overrides(
                dp.Activity,
                values["activity"],
                self.activity,
            )

        try:
            return dp.Product(**values)
        except ValidationError as err:
            raise TraitError(str(err)) from err

    def to_dict(self):
        """Return the configured product overrides as a plain mapping."""
        return {
            "description": self.description,
            "disclaimer": self.disclaimer,
            "instance": dict(self.instance),
            "activity": dict(self.activity),
        }


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

        # Legacy files may contain contact data that does not satisfy the current
        # CTAO model. Preserve the original values here so migration can validate
        # each field individually and replace invalid values with their defaults.
        contact = Contact()
        with contact.cross_validation_lock:
            for key, value in kwargs["contact"].items():
                setattr(contact, key, value)

        return cls(
            contact=contact,
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
    meta = _read_hdf5_attributes(h5file, path)
    if "CTA REFERENCE VERSION" not in meta:
        raise ValueError("No legacy CTA reference metadata found")
    return Reference.from_dict(meta)


def _read_reference_metadata_ecsv(path):
    """Read legacy CTA reference metadata from an ECSV file."""
    return Reference.from_dict(Table.read(path).meta)


def _read_reference_metadata_fits(fitsfile, hdu: int | str = 0):
    """Read legacy reference metadata from a FITS HDU.

    Parameters
    ----------
    fitsfile : path-like or astropy.io.fits.HDUList
        FITS file or an open HDU list.
    hdu : int or str
        HDU index or name.

    Returns
    -------
    Reference
        Parsed legacy reference metadata.
    """
    with ExitStack() as stack:
        if not isinstance(fitsfile, fits.HDUList):
            fitsfile = stack.enter_context(fits.open(fitsfile))

        return Reference.from_fits(fitsfile[hdu].header)


# -----------------------------------------------------
#  New Data Model
# -----------------------------------------------------


class LegacyMetadataWarning(CTAPipeDeprecationWarning):
    """Warning emitted when deprecated legacy CTA metadata is encountered."""


def get_compatible_metadata_versions(
    current_version=None,
) -> set[str]:
    """Return metadata versions that can be migrated to the current version.

    Compatibility is determined from the migration history provided by
    ``ctao_datamodel``. A version is considered compatible if there is a complete
    migration path from that version to ``current_version``.

    Parameters
    ----------
    current_version : str, optional
        Target metadata version. If omitted, the current
        ``dp.Product.ctao_metadata_version`` is used.

    Returns
    -------
    set[str]
        Metadata versions that can be migrated to the target version, including
        the target version itself.
    """
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


def _check_metadata_version(version: str | None) -> None:
    """Check that the CTAO metadata version is supported."""
    if version is None:
        raise ValueError("Unsupported metadata format")

    if version not in get_compatible_metadata_versions():
        raise ValueError(f"Unsupported CTAO metadata version: {version}")


def read_ctao_metadata(input_file: str | Path | tables.File) -> dp.Product:
    """Read CTAO product metadata from a supported file.

    The format is detected from the file contents. FITS (including gzip-compressed
    FITS), HDF5, ECSV, and JSON are supported. Both current CTAO metadata and legacy
    ctapipe reference metadata are supported for all file formats. Legacy metadata is
    converted to the current CTAO data model and emits a
    :class:`~ctapipe.io.metadata.LegacyMetadataWarning`.

    Parameters
    ----------
    input_file : path-like or tables.File
        Input file or open PyTables file handle.

    Returns
    -------
    ctao_datamodel.models.dataproducts.Product
        Validated CTAO product metadata.

    Raises
    ------
    ValueError
        If the file format or metadata format is unsupported.
    pydantic.ValidationError
        If the metadata does not validate against the CTAO product model.
    """
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

    if first_bytes.startswith(b"{"):
        return _read_json_metadata(input_file)

    if first_bytes.startswith(b"# %ECSV"):
        return _read_ecsv_metadata(input_file)

    if first_bytes.startswith(b"SIMPLE"):
        return _read_fits_metadata(input_file)

    raise ValueError(
        f"'{input_file}' is not one of the supported file formats: fits, hdf5, ecsv, json"
    )


def _read_ecsv_metadata(ecsv_file) -> dp.Product:
    """Read CTAO product metadata from an ECSV file."""
    metadata = Table.read(ecsv_file).meta

    if "CTAO.ctao_metadata_version" in metadata:
        _check_metadata_version(metadata["CTAO.ctao_metadata_version"])
        return metadata_to_product(metadata)

    if "CTA REFERENCE VERSION" in metadata:
        return _legacy_to_product(metadata)

    raise ValueError("Unsupported metadata format")


def _read_json_metadata(json_file) -> dp.Product:
    """Read CTAO product metadata from a JSON file."""
    import json

    with open(json_file) as f:
        metadata = json.load(f)

    metadata = metadata.get("metadata", metadata)

    if "CTAO.ctao_metadata_version" in metadata:
        _check_metadata_version(metadata["CTAO.ctao_metadata_version"])
        return metadata_to_product(metadata)

    if "CTA REFERENCE VERSION" in metadata:
        return _legacy_to_product(metadata)

    raise ValueError("Unsupported metadata format")


def _read_fits_metadata(fits_file) -> dp.Product:
    """Read current or legacy CTAO product metadata from a FITS file."""
    with fits.open(fits_file) as hdul:
        header = hdul[0].header

        # Current CTAO metadata
        if "CTAOMETA" in header:
            _check_metadata_version(header["CTAOMETA"])

            # Temporary workaround for
            # https://gitlab.cta-observatory.org/cta-computing/common/ctao-datamodel/-/work_items/48
            if "MODEL" in header and "MODELURL" not in header:
                header["MODELURL"] = None
            if "SOFTWARE" in header and "SOFTURL" not in header:
                header["SOFTURL"] = None

            return dm.fits_header_to_instance(
                header,
                model=dp.Product,
            )

        # Legacy CTA metadata
        if "CTA REFERENCE VERSION" in header:
            return _legacy_to_product(header)

        raise ValueError("Unsupported metadata format")


def _read_hdf5_metadata(h5file, path="/") -> dp.Product:
    """Read current or legacy CTAO product metadata from an HDF5 file or node."""
    metadata = _read_hdf5_attributes(h5file, path)

    if "CTAO.ctao_metadata_version" in metadata:
        _check_metadata_version(metadata.get("CTAO.ctao_metadata_version"))
        return metadata_to_product(metadata)

    if "CTA REFERENCE VERSION" in metadata:
        return _legacy_to_product(metadata)

    raise ValueError("Unsupported metadata format")


def _read_hdf5_attributes(h5file, path="/"):
    """Read hdf5 attributes into a dict"""
    with ExitStack() as stack:
        if not isinstance(h5file, tables.File):
            h5file = stack.enter_context(tables.open_file(h5file))

        node = h5file.get_node(path)
        return {key: node._v_attrs[key] for key in node._v_attrs._f_list()}


def _legacy_to_product(metadata) -> dp.Product:
    """Convert legacy CTA reference metadata to a current CTAO product."""
    warnings.warn(
        "Legacy ctapipe metadata detected. "
        "If this file is not already being migrated, use ctapipe-merge to convert it "
        "to the current CTAO metadata format.",
        LegacyMetadataWarning,
        stacklevel=2,
    )

    reference = Reference.from_dict(metadata)

    product_type = _to_ctao_product_type(reference)
    contact = _to_ctao_contact(reference)

    instance_kwargs = {"id": _legacy_uuid(reference.product.id_, "product")}

    # Legacy data levels -> processing sublevel
    data_levels = set(reference.product.data_levels)
    has_dl1_images = DataLevel.DL1_IMAGES in data_levels
    has_dl1_parameters = DataLevel.DL1_PARAMETERS in data_levels
    sublevel = None
    if has_dl1_images:
        sublevel = dp.ProcessingSublevel.IMAGES

    if has_dl1_parameters:
        sublevel = (
            dp.ProcessingSublevel.PARAMETERS
            if sublevel is None
            else sublevel | dp.ProcessingSublevel.PARAMETERS
        )

    if sublevel is not None:
        instance_kwargs["sublevel_id"] = sublevel

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

    elif reference.instrument.class_ == "Subarray" and instrument_id is not None:
        instance_kwargs["subarray_id"] = instrument_id

    model_url = _legacy_optional_string(reference.product.data_model_url)

    return dp.Product(
        description=reference.product.description,
        creation_time=reference.product.creation_time,
        curation=dp.Curation(),
        data=product_type,
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


def _to_ctao_contact(reference: Reference) -> dp.Contact:
    """Convert legacy contact information, falling back field by field."""
    migrated = Contact()
    invalid = {}

    for field in ("name", "organization", "email"):
        value = _legacy_optional_string(getattr(reference.contact, field))

        if value is None:
            continue

        try:
            setattr(migrated, field, value)
        except TraitError:
            invalid[field] = value

    if invalid:
        warnings.warn(
            "Legacy metadata contains invalid contact information: "
            f"{invalid!r}. Using default values for these fields.",
            LegacyMetadataWarning,
            stacklevel=2,
        )

    return migrated.to_model()


def _to_ctao_product_type(reference: Reference) -> dp.ProductType:
    """Derive a current CTAO product type from legacy reference metadata."""
    level = to_ctao_data_level(reference.product.data_levels)

    if level is None:
        warnings.warn(
            "Could not determine a data level from legacy metadata. "
            "Falling back to DataLevel.SIM.",
            LegacyMetadataWarning,
            stacklevel=2,
        )
        level = dp.DataLevel.SIM

    try:
        association = dp.DataAssociation(reference.product.data_association)
    except ValueError:
        association = dp.DataAssociation.SUBARRAY
        warnings.warn(
            "Could not determine a valid data association from legacy metadata "
            f"{reference.product.data_association!r}. "
            "Falling back to DataAssociation.SUBARRAY.",
            LegacyMetadataWarning,
            stacklevel=2,
        )

    data_type = dp.DataType.OBSERVATION_SIM

    if reference.process.type_ == "Observation":
        data_type = dp.DataType.OBSERVATION

    return dp.ProductType(
        level=level,
        division=dp.DataDivision.EVENT,
        association=association,
        type=data_type,
    )


def to_ctao_data_level(data_levels: Iterable[DataLevel]) -> dp.DataLevel | None:
    """Select the primary CTAO data level from ctapipe data levels.

    DL1 sublevels such as images, parameters, and muon data are normalized to
    ``DL1``. If several levels are present, the highest CTAO data level is returned.

    Parameters
    ----------
    data_levels : collections.abc.Iterable of DataLevel
        ctapipe data levels to convert.

    Returns
    -------
    ctao_datamodel.models.dataproducts.DataLevel or None
        Primary CTAO data level, or ``None`` if ``data_levels`` is empty.
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
        return None

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


def _legacy_site_id(site: str | None) -> SiteID | None:
    """Convert an unambiguous legacy instrument site to a CTAO SiteID."""
    site = _legacy_optional_string(site)
    site_default = dp.InstanceIdentifier.model_fields["site_id"].default
    if site is None:
        return site_default

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

    return mapping.get(site, site_default)


def _legacy_instrument_id(value: str | None) -> int | None:
    """Convert a legacy instrument id to an integer if possible."""
    value = _legacy_optional_string(value)
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return None


def metadata_to_product(metadata) -> dp.Product:
    """Convert flattened current CTAO metadata into a validated product model.

    Parameters
    ----------
    metadata : collections.abc.Mapping
        Flattened metadata with keys below the ``CTAO`` namespace. Unrelated
        entries are ignored.

    Returns
    -------
    ctao_datamodel.models.dataproducts.Product
        Validated product metadata, migrated to the current metadata version
        when supported by ``ctao-datamodel``.
    """
    metadata = {
        key: value for key, value in metadata.items() if key.startswith("CTAO.")
    }

    # Temporary workaround for
    # https://gitlab.cta-observatory.org/cta-computing/common/ctao-datamodel/-/work_items/48
    # Only add optional URL fields when their parent metadata object already exists.
    if any(key.startswith("CTAO.model.") for key in metadata):
        metadata.setdefault("CTAO.model.url", None)

    if any(key.startswith("CTAO.activity.software.") for key in metadata):
        metadata.setdefault("CTAO.activity.software.url", None)

    return dm.unflatten_model_instance(
        metadata,
        model=dp.Product,
        parent_key="CTAO",
    )


def activity_from_provenance(activity) -> dp.Activity:
    """Create CTAO activity metadata from a ctapipe provenance activity.

    Parameters
    ----------
    activity : ctapipe.core.provenance._ActivityProvenance
        Active or completed provenance activity.

    Returns
    -------
    ctao_datamodel.models.dataproducts.Activity
        Activity metadata containing the process name, identifier, timestamps,
        and ctapipe software information.
    """
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
            url="https://github.com/cta-observatory/ctapipe",
        ),
        configuration_id="",
    )


def write_product_metadata_hdf5(
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

    # Remove current metadata first
    node = h5file.get_node(path)
    for name in node._v_attrs._f_list("user"):
        if name.startswith("CTAO."):
            del node._v_attrs[name]

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


def write_product_metadata_fits_header(
    product: dp.Product,
    header: fits.Header,
):
    """Write CTAO product metadata to a FITS header.

    Parameters
    ----------
    product : ctao_datamodel.models.dataproducts.Product
        Validated product metadata to serialize.
    header : astropy.io.fits.Header
        Header updated in place with CTAO metadata keywords.
    """
    metadata = dm.instance_to_fits_header(product)
    header.update(metadata)
