"""
Dump instrumental descriptions in a monte-carlo (simtelarray) input file to
FITS files that can be loaded independently (e.g. with
CameraGeometry.from_table()).  The name of the output files are
automatically generated.
"""

import os
import pathlib

import ctao_datamodel as dm
import ctao_datamodel.models.dataproducts as dp
from astropy.time import Time
from ctao_datamodel.models.common import SiteID

from ..compat import ECSV_FMT
from ..core import Provenance, Tool
from ..core.traits import Dict, Enum, Path, Unicode
from ..exceptions import InputMissing
from ..io import EventSource

__all__ = ["DumpInstrumentTool"]


class DumpInstrumentTool(Tool):
    description = Unicode(__doc__)
    name = "ctapipe-dump-instrument"

    outdir = Path(
        file_ok=False,
        directory_ok=True,
        allow_none=True,
        default_value=None,
        help="Output directory. If not given, the current working directory will be used.",
    ).tag(config=True)

    format = Enum(
        ["fits", "ecsv", "hdf5", "service"],
        default_value="fits",
        help="Format of output file. 'service' creates CTAO service data format directory structure.",
        config=True,
    )
    contact = Dict(
        default_value={
            "name": "unknown",
            "email": "unknown@example.org",
            "organization": "unknown",
        },
        help="Contact information for generated data products.",
    ).tag(config=True)

    aliases = {
        ("i", "input"): "EventSource.input_url",
        ("f", "format"): "DumpInstrumentTool.format",
        ("o", "outdir"): "DumpInstrumentTool.outdir",
    }

    classes = [EventSource]

    def setup(self):
        try:
            with EventSource(parent=self) as source:
                self.infile = source.input_url
                self.subarray = source.subarray
                self.is_simulation = source.is_simulation
        except InputMissing:
            self.log.critical(
                "Specifying EventSource.input_url is required (via -i, --input or a config file)."
            )
            self.exit(1)

    def start(self):
        if self.outdir is None:
            self.outdir = pathlib.Path(os.getcwd())

        self.outdir.mkdir(exist_ok=True, parents=True)

        if self.format == "hdf5":
            self.subarray.to_hdf(self.outdir / "subarray.h5")
        elif self.format == "service":
            self.write_service_data()
        else:
            self.write_camera_definitions()
            self.write_optics_descriptions()
            self.write_subarray_description()

    def finish(self):
        pass

    @staticmethod
    def _get_file_format_info(format_name):
        """returns file extension + dict of required parameters for
        Table.write"""
        if format_name == "fits":
            return "fits.gz", dict()
        elif format_name == "ecsv":
            return "ecsv", dict()
        else:
            raise NameError(f"format {format_name} not supported")

    def write_camera_definitions(self):
        """writes out camgeom and camreadout files for each camera"""
        self.subarray.info(printer=self.log.info)
        for camera in self.subarray.camera_types:
            self.write_single_camera(camera)

    def write_single_camera(self, camera, outdir=None, name_prefix=None):
        """Write out camera geometry and readout for a single camera.

        Parameters
        ----------
        camera : CameraDescription
            CameraDescription object to write out
        outdir : Path, optional
            Directory to write files to. If None, uses self.outdir
        name_prefix : str, optional
            Prefix for output filenames. If None, uses camera.name
        """
        outdir = self.outdir if outdir is None else outdir
        name_prefix = camera.name if name_prefix is None else name_prefix
        ext, args = self._get_file_format_info(self.format)
        self.log.debug("Writing camera %s", camera)
        geom = camera.geometry
        readout = camera.readout

        geom_table = geom.to_table()
        geom_table.meta["SOURCE"] = str(self.infile)
        geom_filename = outdir / f"{name_prefix}.camgeom.{ext}"

        readout_table = readout.to_table()
        readout_table.meta["SOURCE"] = str(self.infile)
        readout_filename = outdir / f"{name_prefix}.camreadout.{ext}"

        try:
            geom_table.write(geom_filename, **args)
            Provenance().add_output_file(geom_filename, "CameraGeometry")
        except OSError as err:
            self.log.exception("couldn't write camera geometry because: %s", err)

        try:
            readout_table.write(readout_filename, **args)
            Provenance().add_output_file(readout_filename, "CameraReadout")
        except OSError as err:
            self.log.exception("couldn't write camera definition because: %s", err)

    def write_optics_descriptions(self):
        """writes out optics files for each telescope type"""
        sub = self.subarray
        ext, args = self._get_file_format_info(self.format)

        tab = sub.to_table(kind="optics")
        tab.meta["SOURCE"] = str(self.infile)
        filename = self.outdir / f"{sub.name}.optics.{ext}"
        try:
            tab.write(filename, **args)
            Provenance().add_output_file(filename, "OpticsDescription")
        except OSError as err:
            self.log.exception(
                "couldn't write optics description '%s' because: %s", filename, err
            )

    def write_subarray_description(self):
        sub = self.subarray
        ext, args = self._get_file_format_info(self.format)
        tab = sub.to_table(kind="subarray", meta_convention="fits")
        tab.meta["SOURCE"] = str(self.infile)
        filename = self.outdir / f"{sub.name}.subarray.{ext}"
        try:
            tab.write(filename, **args)
            Provenance().add_output_file(filename, "SubarrayDescription")
        except OSError as err:
            self.log.exception(
                "couldn't write subarray description '%s' because: %s", filename, err
            )

    def _create_service_product(
        self,
        description,
        model_name,
        model_version,
        site,
        subarray_id,
        model_url=None,
    ):
        from ctapipe.io import metadata as meta

        activity = Provenance().current_activity

        return dp.Product(
            description=description,
            creation_time=Time.now(),
            data=dp.ProductType(
                level=dp.DataLevel.DL0,
                division=dp.DataDivision.SERVICE,
                association=dp.DataAssociation.SUBARRAY,
                type=(
                    dp.DataType.OBSERVATION_SIM
                    if self.is_simulation
                    else dp.DataType.OBSERVATION
                ),
            ),
            instance=dp.InstanceIdentifier(
                site_id=SiteID(site),
                subarray_id=subarray_id,
            ),
            curation=dp.Curation(),
            model=dp.DataModel(
                name=model_name,
                version=model_version,
                url=model_url,
            ),
            contact=dp.Contact(**self.contact),
            activity=(
                meta.activity_from_provenance(activity)
                if activity is not None
                else None
            ),
        )

    @staticmethod
    def _flatten_product(product):
        return dm.flatten_model_instance(
            product,
            parent_key="CTAO",
        )

    def write_service_data(self, subarray_id=1, site=None):
        """
        Write SubarrayDescription to service data directory structure.

        This creates the directory structure and files, which can later be loaded with
        `~ctapipe.instrument.SubarrayDescription.from_service_data`.

        Parameters
        ----------
        subarray_id : int, optional
            Subarray ID to assign (default: 1)
        site : str, optional
            Site name (e.g., "CTAO-North", "CTAO-South").
            If not provided, it will be inferred from the reference location.
        """
        import json

        from astropy.table import QTable

        sub = self.subarray
        self.outdir.mkdir(exist_ok=True, parents=True)

        # Create instrument directory to match CTAO service data structure
        instrument_dir = self.outdir / "instrument"
        instrument_dir.mkdir(exist_ok=True, parents=True)

        self.log.info(
            "Writing instrument description in CTAO service data format to %s",
            self.outdir,
        )

        # Infer site from coordinates if not provided
        if site is None:
            lat = sub.reference_location.geodetic.lat.value
            site = "CTAO-North" if lat > 0 else "CTAO-South"

        try:
            site = SiteID(site).value
        except ValueError as err:
            raise ValueError(
                f"Invalid site {site!r}, expected one of "
                f"{[site.value for site in SiteID]}"
            ) from err

        # instrument.meta.json
        instrument_product = self._create_service_product(
            description=f"Instrument description for {sub.name}",
            model_name="CTAO Service Data",
            model_version=sub.CURRENT_SERVICE_DATA_VERSION,
            site=site,
            subarray_id=subarray_id,
        )

        meta_file = instrument_dir / "instrument.meta.json"
        with open(meta_file, "w") as f:
            json.dump(self._flatten_product(instrument_product), f, indent=2)

        Provenance().add_output_file(meta_file, "ServiceDataMeta")

        # array-element-ids.json
        ae_product = self._create_service_product(
            description=f"Array element IDs for {sub.name}",
            model_name="ctao.common.identifiers.array_elements",
            model_version=sub.CURRENT_ARRAY_ELEMENTS_IDENTIFIERS_VERSION,
            model_url="https://gitlab.cta-observatory.org/cta-computing/common/identifiers",
            site=site,
            subarray_id=subarray_id,
        )

        array_element_ids = {
            "metadata": self._flatten_product(ae_product),
            "array_elements": [
                {"id": int(tel_id), "name": f"TEL{tel_id:03d}"}
                for tel_id, tel in sub.tels.items()
            ],
        }
        ae_ids_file = instrument_dir / "array-element-ids.json"
        with open(ae_ids_file, "w") as f:
            json.dump(array_element_ids, f, indent=2)
        Provenance().add_output_file(ae_ids_file, "ServiceDataArrayElements")

        # subarray-ids.json
        subarray_product = self._create_service_product(
            description=f"Subarray IDs for {sub.name}",
            model_name="ctao.common.identifiers.subarrays",
            model_version=sub.CURRENT_SUBARRAY_IDENTIFIERS_VERSION,
            model_url="https://gitlab.cta-observatory.org/cta-computing/common/identifiers",
            site=site,
            subarray_id=subarray_id,
        )

        subarray_ids = {
            "metadata": self._flatten_product(subarray_product),
            "subarrays": [
                {
                    "id": subarray_id,
                    "name": sub.name,
                    "site": site,
                    "array_element_ids": [int(tel_id) for tel_id in sub.tel_ids],
                }
            ],
        }
        subarray_ids_file = instrument_dir / "subarray-ids.json"
        with open(subarray_ids_file, "w") as f:
            json.dump(subarray_ids, f, indent=2)
        Provenance().add_output_file(subarray_ids_file, "ServiceDataSubarrays")

        # Create positions directory and file
        positions_dir = instrument_dir / "positions"
        positions_dir.mkdir(exist_ok=True)

        # Get reference location in ITRS coordinates
        itrs = sub.reference_location.itrs

        # Create positions table
        positions_table = QTable(
            {
                "ae_id": [int(tel_id) for tel_id in sub.tel_ids],
                "name": [tel.name for tel in sub.tels.values()],
                "x": [sub.positions[tel_id][0] for tel_id in sub.tel_ids],
                "y": [sub.positions[tel_id][1] for tel_id in sub.tel_ids],
                "z": [sub.positions[tel_id][2] for tel_id in sub.tel_ids],
            }
        )
        positions_table.meta["reference_x"] = str(itrs.x)
        positions_table.meta["reference_y"] = str(itrs.y)
        positions_table.meta["reference_z"] = str(itrs.z)
        positions_table.meta["site"] = site
        positions_product = self._create_service_product(
            description=f"Array element positions for {sub.name}",
            model_name="CTAO Service Data",
            model_version=sub.CURRENT_SERVICE_DATA_VERSION,
            site=site,
            subarray_id=subarray_id,
        )

        positions_table.meta.update(self._flatten_product(positions_product))

        positions_file = (
            positions_dir / f"{site.replace(' ', '_')}_ArrayElementPositions.ecsv"
        )
        positions_table.write(positions_file, format=ECSV_FMT, overwrite=True)
        Provenance().add_output_file(positions_file, "ServiceDataPositions")

        # Group telescopes by unique TelescopeDescription.
        # str(tel_desc) = "{size_type}_{optics_name}_{camera_name}" and is used
        # as the shared type-directory name.
        array_elements_dir = instrument_dir / "array-elements"
        array_elements_dir.mkdir(exist_ok=True, parents=True)

        # Write shared files once per unique telescope type
        for index, tel_desc in enumerate(sub.telescope_types):
            type_key = f"TEL-TYPE-{index:02d}"
            type_dir = array_elements_dir / type_key
            type_dir.mkdir(exist_ok=True, parents=True)
            self.log.debug("Writing shared type directory %s", type_key)

            optics_file = type_dir / f"{type_key}.tel_optics.ecsv"
            optics_table = tel_desc.optics.to_table()
            optics_table.write(optics_file, format=ECSV_FMT, overwrite=True)
            Provenance().add_output_file(optics_file, "ServiceDataOptics")

            orig_format = self.format
            self.format = "fits"
            try:
                self.write_single_camera(
                    tel_desc.camera, outdir=type_dir, name_prefix=type_key
                )
            finally:
                self.format = orig_format

        # Create a real directory for each array element with per-file symlinks
        # pointing to the shared type directory (deduplication).
        for tel_id, tel_desc in sub.tel.items():
            ae_id_str = f"{tel_id:03d}"
            ae_dir = array_elements_dir / ae_id_str
            ae_dir.mkdir(exist_ok=True, parents=True)
            index = sub.telescope_types.index(tel_desc)
            type_key = f"TEL-TYPE-{index:02d}"
            self.log.debug(
                "Writing array element %s -> %s (symlinks)", ae_id_str, type_key
            )
            for suffix in ["tel_optics.ecsv", "camgeom.fits.gz", "camreadout.fits.gz"]:
                link = ae_dir / f"{ae_id_str}.{suffix}"
                link.symlink_to(f"../{type_key}/{type_key}.{suffix}")

        self.log.info("Service data written successfully to %s", self.outdir)


def main():
    tool = DumpInstrumentTool()
    tool.run()


if __name__ == "__main__":
    main()
