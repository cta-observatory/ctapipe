import hashlib
import json
import zipfile
from io import TextIOWrapper

import ctao_datamodel as dm
import ctao_datamodel.models.dataproducts as dp
import joblib
import tables
from astropy.time import Time
from astropy.utils import lazyproperty

from ..core import Component, Provenance
from ..core.provenance import json_config_handler
from ..core.traits import Bool, Int, Path, TraitError
from ..exceptions import InputMissing
from ..instrument import SubarrayDescription
from .metadata import (
    Contact,
    Curation,
    ProductMetadata,
    activity_from_provenance,
    read_ctao_metadata,
)

BAGIT_PAYLOAD_PREFIX = "data"
METADATA_NAME = "bag-info.txt"
SUBARRAY_NAME = f"{BAGIT_PAYLOAD_PREFIX}/subarray.h5"
RECONSTRUCTOR_CONFIG_NAME = f"{BAGIT_PAYLOAD_PREFIX}/reconstructor_config.json"

CURRENT_VERSION = "1.0.0"
SUPPORTED_VERSIONS = {CURRENT_VERSION}


def _joblib_name(key):
    key = str(key)
    if (
        not key
        or "\\" in key
        or "\0" in key
        or any(part in {"", ".", ".."} for part in key.split("/"))
    ):
        raise ValueError(f"Invalid model key: {key!r}")
    return f"{BAGIT_PAYLOAD_PREFIX}/{key}.pkl"


class BagItWriter(Component):
    """
    Write multiple models, e.g. sklearn models, into a BagIt 1.0 ZIP archive.

    Models, reconstructor configuration and the subarray are stored under
    ``data/``. CTAO product metadata is stored as dot-separated keys in
    ``bag-info.txt`` at the bag root.
    SHA-256 payload and tag manifests are written when the archive is closed.
    """

    output_path = Path(directory_ok=False, help="Output path").tag(config=True)

    compression_level = Int(
        default_value=9,
        help="Compression level passed to joblib when storing objects",
    ).tag(config=True)

    overwrite = Bool(
        default_value=False,
        help=(
            "If True, overwrite an existing output_path"
            "if False an error will be raised if the output_path already exists"
        ),
    ).tag(config=True)

    def __init__(self, output_path=None, **kwargs):
        if output_path is not None:
            kwargs["output_path"] = output_path
        super().__init__(**kwargs)

        if self.output_path is None:
            raise TraitError("output_path of BagItWriter must not be None")

        if not self.overwrite and self.output_path.exists():
            raise TraitError(
                f"Output path {self.output_path} exists and overwrite=False"
            )

        self.contact_info = Contact(parent=self)
        self.curation_info = Curation(parent=self)
        self.product_info = ProductMetadata(parent=self)

        self.output_path.parent.mkdir(exist_ok=True, parents=True)
        self.outfile = zipfile.ZipFile(
            self.output_path,
            mode="w",
            compresslevel=self.compression_level,
            # we store the model payloads zstd compressed, so no additional compression
            compression=zipfile.ZIP_STORED,
        )
        self.outfile.mkdir(BAGIT_PAYLOAD_PREFIX)
        self.outfile.writestr(
            "bagit.txt",
            "BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n",
        )

    def close(self):
        if self.outfile.fp is None:
            return

        try:
            self._write_product_meta()
            self._write_manifest(payload=True)
            self._write_manifest(payload=False)
        finally:
            self.outfile.close()
            Provenance().add_output_file(
                self.output_path, role="dl2/service/models", add_meta=True
            )

    @staticmethod
    def _escape_manifest_filename(name):
        # BagIt requires escaping these characters in manifest paths.
        return name.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")

    def _write_manifest(self, payload):
        # two manifests, one for the payload files in data/, one for the toplevel
        hashfunc = "sha256"

        lines = []
        payload_prefix = f"{BAGIT_PAYLOAD_PREFIX}/"
        for info in self.outfile.infolist():
            if info.is_dir() or info.filename.startswith(payload_prefix) != payload:
                continue

            with self.outfile.open(info) as f:
                checksum = hashlib.file_digest(f, hashfunc).hexdigest()

            filename = self._escape_manifest_filename(info.filename)
            lines.append(f"{checksum}  {filename}\n")

        prefix = "" if payload else "tag"
        name = f"{prefix}manifest-{hashfunc}.txt"
        self.outfile.writestr(name, "".join(lines).encode("utf-8"))

    def _get_product_meta(self):
        activity = Provenance().current_activity
        activity_metadata = (
            {"activity": activity_from_provenance(activity)}
            if activity is not None
            else {}
        )
        return self.product_info.to_model(
            creation_time=Time.now(),
            data=dp.ProductType(
                level=dp.DataLevel.DL2,
                division=dp.DataDivision.SERVICE,
                association=dp.DataAssociation.SUBARRAY,
                type=dp.DataType.RECO_MODEL,
            ),
            instance=dp.InstanceIdentifier(),
            model=dp.DataModel(
                name="ctapipe/dl2/service/model",
                version=CURRENT_VERSION,
            ),
            contact=self.contact_info.to_model(),
            curation=self.curation_info.to_model(),
            **activity_metadata,
        )

    def _write_product_meta(self):
        metadata = dm.flatten_model_instance(self._get_product_meta(), separator=".")
        lines = []
        for key, value in metadata.items():
            # BagIt folds long/multiline values onto indented continuation lines.
            value = "\n ".join(value.splitlines())
            lines.append(f"{key}: {value}\n")
        self.outfile.writestr(METADATA_NAME, "".join(lines).encode("utf-8"))

    def _check_exists(self, name):
        if name in self.outfile.namelist():
            raise FileExistsError(
                f"A file with name {name!r} already exists in the archive."
            )

    def write_joblib(self, key, obj):
        name = _joblib_name(key)
        self._check_exists(name)

        with self.outfile.open(name, mode="w") as f:
            joblib.dump(obj, f, compress=self.compression_level)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write_subarray(self, subarray):
        name = SUBARRAY_NAME
        self._check_exists(name)

        h5file = tables.open_file(
            "__in_memory__",
            mode="w",
            # with these options, the file is created in memory only
            driver="H5FD_CORE",
            driver_core_backing_store=0,
        )
        subarray.to_hdf(h5file)
        h5file.flush()
        hdf5_payload = h5file.get_file_image()
        h5file.close()

        with self.outfile.open(name, "w") as f:
            f.write(hdf5_payload)

    def _write_reconstructor_config(self, reconstructor):
        name = RECONSTRUCTOR_CONFIG_NAME
        self._check_exists(name)

        config = reconstructor.get_current_config()

        with (
            self.outfile.open(name, mode="w") as f,
            TextIOWrapper(f, encoding="utf-8") as wrapper,
        ):
            json.dump(config, wrapper, default=json_config_handler)

    def write_reconstructor(self, reconstructor):
        self.write_joblib("reconstructor", reconstructor)
        self._write_reconstructor_config(reconstructor)
        self.write_subarray(reconstructor.subarray)


class BagItReader(Component):
    input_path = Path().tag(config=True)

    def __init__(self, input_path=None, **kwargs):
        if input_path is not None:
            kwargs["input_path"] = input_path
        super().__init__(**kwargs)

        if self.input_path is None:
            raise InputMissing("input_path")

        self.zip = zipfile.ZipFile(self.input_path, mode="r")

        self._files = self.zip.namelist()
        self._keys = tuple(
            f.removeprefix(f"{BAGIT_PAYLOAD_PREFIX}/").removesuffix(".pkl")
            for f in self._files
            if f.startswith(f"{BAGIT_PAYLOAD_PREFIX}/")
            and f.endswith(".pkl")
            and f != f"{BAGIT_PAYLOAD_PREFIX}/reconstructor.pkl"
        )

        try:
            self._meta = read_ctao_metadata(self.zip)
            if self._meta.model.version not in SUPPORTED_VERSIONS:
                raise ValueError(
                    f"Unsupported model data format version: {self._meta.model.version}"
                )
        except Exception:
            self.zip.close()
            raise

    @property
    def keys(self):
        return self._keys

    @property
    def meta(self):
        """Validated CTAO product metadata."""
        return self._meta

    @lazyproperty
    def subarray(self):
        if SUBARRAY_NAME not in self._files:
            raise KeyError(f"No subarray stored in input_path {self.input_path}")

        with self.zip.open(SUBARRAY_NAME, "r") as f:
            hdf5_payload = f.read()
        with tables.open_file(
            # must not exist on disk, h5 still checks even when using the in-memory data
            "__in_memory__",
            mode="r",
            # with these options, the file is created in memory only
            driver="H5FD_CORE",
            driver_core_image=hdf5_payload,
            driver_core_backing_store=0,
        ) as h5file:
            return SubarrayDescription.from_hdf(h5file)

    def read_joblib(self, key):
        with self.zip.open(_joblib_name(key), "r") as f:
            return joblib.load(f)

    def read_reconstructor(self):
        reconstructor = self.read_joblib("reconstructor")
        reconstructor.subarray = self.subarray
        return reconstructor

    def close(self):
        self.zip.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
