import json
import zipfile
from io import TextIOWrapper

import msgpack
import tables
import zstandard
from sklearn_migrator import serialize

from ..core import Component
from ..core.provenance import json_config_handler
from ..core.traits import Int, Path
from .metadata import Activity, Contact, Instrument, Process, Product, Reference


class SklearnModelWriter(Component):
    """
    Writer for sklearn models.

    Uses a compressed tarfile to write multiple models to the same
    file. Models are serialized using sklearn-migrator.
    """

    output_path = Path(directory_ok=False, help="Output path").tag(config=True)
    compression_level = Int(default_value=9).tag(config=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.outfile = zipfile.ZipFile(
            self.output_path,
            mode="x",
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
            product=Product(),
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write_subarray(self, subarray):
        h5file = tables.open_file(
            "subarray.h5",
            mode="w",
            # with these options, the file is created in memory only
            driver="H5FD_CORE",
            drive_core_backing_store=0,
        )
        subarray.to_hdf(h5file)
        h5file.flush()
        hdf5_payload = h5file.get_file_image()
        h5file.close()

        with self.outfile.open("subarray.h5", "w") as f:
            f.write(hdf5_payload)

    def write_reconstructor_config(self, reconstructor):
        config = reconstructor.get_current_config()

        with self.outfile.open("reconstructor_config.json", mode="w") as f:
            with TextIOWrapper(f, encoding="utf-8") as wrapper:
                json.dump(config, wrapper, default=json_config_handler)

    def __call__(self, key, model):
        with self.outfile.open(f"{key}.msgpack.zst", "w") as raw:
            cctx = zstandard.ZstdCompressor(level=self.compression_level)
            with cctx.stream_writer(raw) as compressor:
                msgpack.dump(serialize(model), compressor)
