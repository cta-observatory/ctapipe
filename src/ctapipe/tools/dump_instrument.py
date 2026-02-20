"""
Dump instrumental descriptions in a monte-carlo (simtelarray) input file to
FITS files that can be loaded independently (e.g. with
CameraGeometry.from_table()).  The name of the output files are
automatically generated.
"""

import os
import pathlib

from ctapipe.core import Provenance, Tool
from ctapipe.core.traits import Enum, Path, Unicode
from ctapipe.exceptions import InputMissing
from ctapipe.io import EventSource

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
        service_dir = self.outdir / "instrument"
        service_dir.mkdir(exist_ok=True, parents=True)

        self.log.info(
            "Writing instrument description in CTAO service data format to %s",
            service_dir,
        )

        # Infer site from coordinates if not provided
        if site is None:
            # Simple heuristic based on latitude
            lat = sub.reference_location.geodetic.lat.value
            if lat > 0:
                site = "CTAO-North"
            else:
                site = "CTAO-South"

        # Create instrument.meta.json
        instrument_meta = {
            "version": sub.CURRENT_SERVICE_DATA_VERSION,
            "format": "CTAO Service Data",
            "description": f"Instrument description for {sub.name}",
        }
        meta_file = service_dir / "instrument.meta.json"
        with open(meta_file, "w") as f:
            json.dump(instrument_meta, f, indent=2)
        Provenance().add_output_file(meta_file, "ServiceDataMeta")

        # Create array-element-ids.json
        array_element_ids = {
            "metadata": {
                "$schema": "https://gitlab.cta-observatory.org/cta-computing/common/identifiers/-/raw/main/array-element-ids.schema.json",
                "version": "1.0",
            },
            "array_elements": [
                {"id": int(tel_id), "name": tel.name}
                for tel_id, tel in sub.tels.items()
            ],
        }
        ae_ids_file = service_dir / "array-element-ids.json"
        with open(ae_ids_file, "w") as f:
            json.dump(array_element_ids, f, indent=2)
        Provenance().add_output_file(ae_ids_file, "ServiceDataArrayElements")

        # Create subarray-ids.json
        subarray_ids = {
            "metadata": {
                "$schema": "https://gitlab.cta-observatory.org/cta-computing/common/identifiers/-/raw/main/subarray-ids.schema.json",
                "version": "1.0",
            },
            "subarrays": [
                {
                    "id": subarray_id,
                    "name": sub.name,
                    "site": site,
                    "array_element_ids": [int(tel_id) for tel_id in sub.tel_ids],
                }
            ],
        }
        subarray_ids_file = service_dir / "subarray-ids.json"
        with open(subarray_ids_file, "w") as f:
            json.dump(subarray_ids, f, indent=2)
        Provenance().add_output_file(subarray_ids_file, "ServiceDataSubarrays")

        # Create positions directory and file
        positions_dir = service_dir / "positions"
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

        positions_file = (
            positions_dir / f"{site.replace(' ', '_')}_ArrayElementPositions.ecsv"
        )
        positions_table.write(positions_file, format="ascii.ecsv", overwrite=True)
        Provenance().add_output_file(positions_file, "ServiceDataPositions")

        # Create array-elements directory
        array_elements_dir = service_dir / "array-elements"
        array_elements_dir.mkdir(exist_ok=True)

        # Write files for each telescope (using ae_id as directory name)
        for tel_id, tel_desc in sub.tels.items():
            ae_id_str = f"{tel_id:03d}"
            ae_dir = array_elements_dir / ae_id_str
            ae_dir.mkdir(exist_ok=True)

            type_name = tel_desc.optics.name
            self.log.debug(
                "Writing array element %s (%s) to %s", ae_id_str, type_name, ae_dir
            )

            # Write optics file
            optics_table = QTable()
            optics = tel_desc.optics
            optics_table.meta["optics_name"] = optics.name
            optics_table.meta["size_type"] = optics.size_type.value
            optics_table.meta["reflector_shape"] = optics.reflector_shape.value
            optics_table.meta["n_mirrors"] = optics.n_mirrors
            optics_table.meta["equivalent_focal_length"] = str(
                optics.equivalent_focal_length
            )
            optics_table.meta["effective_focal_length"] = str(
                optics.effective_focal_length
            )
            optics_table.meta["mirror_area"] = str(optics.mirror_area)
            optics_table.meta["n_mirror_tiles"] = optics.n_mirror_tiles

            optics_file = ae_dir / f"{ae_id_str}.optics.ecsv"
            optics_table.write(optics_file, format="ascii.ecsv", overwrite=True)
            Provenance().add_output_file(optics_file, "ServiceDataOptics")

            # Write camera geometry and readout files
            # Temporarily set format to 'fits' for service data
            orig_format = self.format
            self.format = "fits"
            try:
                self.write_single_camera(
                    tel_desc.camera, outdir=ae_dir, name_prefix=ae_id_str
                )
            finally:
                self.format = orig_format

        self.log.info("Service data written successfully to %s", service_dir)


def main():
    tool = DumpInstrumentTool()
    tool.run()


if __name__ == "__main__":
    main()
