import json
import zipfile
from io import TextIOWrapper

import joblib
import tables
from astropy.utils import lazyproperty

from ..core import Component
from ..core.provenance import json_config_handler
from ..core.traits import Bool, Int, Path, TraitError
from ..exceptions import InputMissing
from ..instrument import SubarrayDescription
from .metadata import Activity, Contact, Instrument, Process, Product, Reference


class ZipModelWriter(Component):
    """
    Writer for writing multiple models, e.g. sklearn models, into a .zip file.
    """

    output_path = Path(directory_ok=False, help="Output path").tag(config=True)
    compression_level = Int(default_value=9).tag(config=True)
    overwrite = Bool(default_value=False).tag(config=True)

    def __init__(self, output_path=None, **kwargs):
        if output_path is not None:
            kwargs["output_path"] = output_path
        super().__init__(**kwargs)

        if self.output_path is None:
            raise TraitError("output_path of ZipModelWriter must not be None")

        if not self.overwrite and self.output_path.exists():
            raise ValueError(f"output_path={output_path} exists and overwrite=False")

        self.output_path.parent.mkdir(exist_ok=True, parents=True)
        self.outfile = zipfile.ZipFile(
            self.output_path,
            mode="w",
            compresslevel=self.compression_level,
            # we store the model payloads zstd compressed, so no additional compression
            compression=zipfile.ZIP_STORED,
        )

    def close(self):
        self._write_product_meta()
        self.outfile.close()

    def _get_product_meta(self):
        return Reference(
            contact=Contact(),
            product=Product(
                description="ctapipe-trained sklearn machine learning models",
                data_model_name="ctapipe/dl2/service/model",
                data_model_version="1.0.0",
            ),
            process=Process(),
            activity=Activity(),
            instrument=Instrument(),
        ).to_dict()

    def _write_product_meta(self):
        with self.outfile.open("meta.json", mode="w") as f:
            with TextIOWrapper(f, encoding="utf-8") as wrapper:
                json.dump(
                    self._get_product_meta(), wrapper, default=json_config_handler
                )

    def _check_exists(self, name):
        if self.overwrite:
            return

        if name in self.outfile.namelist():
            raise FileExistsError(
                f"A file with name {name!r} already exists in the archive."
            )

    def _write_compressed_joblib(self, key, obj):
        name = f"{key}.pkl"
        self._check_exists(name)

        with self.outfile.open(name, mode="w") as f:
            joblib.dump(obj, f, compress=self.compression_level)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write_subarray(self, subarray):
        name = "subarray.h5"
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
        name = "reconstructor_config.json"
        self._check_exists(name)

        config = reconstructor.get_current_config()

        with self.outfile.open(name, mode="w") as f:
            with TextIOWrapper(f, encoding="utf-8") as wrapper:
                json.dump(config, wrapper, default=json_config_handler)

    def write_reconstructor(self, reconstructor):
        self._write_compressed_joblib("reconstructor", reconstructor)
        self._write_reconstructor_config(reconstructor)
        self.write_subarray(reconstructor.subarray)

    def __call__(self, key, model):
        self._write_compressed_joblib(key, model)


SUPPORTED_VERSIONS = {"1.0.0"}


class ZipModelReader(Component):
    input_path = Path().tag(config=True)

    def __init__(self, input_path=None, **kwargs):
        super().__init__(input_path=input_path, **kwargs)

        if self.input_path is None:
            raise InputMissing("input_path")

        self.zip = zipfile.ZipFile(self.input_path, mode="r")

        self._files = self.zip.namelist()
        if "meta.json" not in self._files:
            raise ValueError(f"input_path {self.input_path} does not contain meta.json")

        self._keys = tuple(
            f.removesuffix(".pkl") for f in self._files if f.endswith(".pkl")
        )

        self._meta = json.loads(self.zip.read("meta.json").decode("utf-8"))
        if self._meta["CTA PRODUCT DATA MODEL VERSION"] not in SUPPORTED_VERSIONS:
            raise ValueError("")

    @property
    def keys(self):
        return self._keys

    @property
    def meta(self):
        return self._keys

    @lazyproperty
    def subarray(self):
        if "subarray.h5" not in self._files:
            raise KeyError(f"No subarray stored in input_path {self.input_path}")

        with self.zip.open("subarray.h5", "r") as f:
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

    def _read_compressed_joblib(self, key):
        with self.zip.open(f"{key}.pkl", "r") as f:
            return joblib.load(f)

    def read_reconstructor(self):
        reconstructor = self._read_compressed_joblib("reconstructor")
        reconstructor.subarray = self.subarray
        return reconstructor

    def close(self):
        self.zip.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
