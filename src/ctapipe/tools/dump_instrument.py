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
from ctapipe.io import EventSource


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
        ["fits", "ecsv", "hdf5"],
        default_value="fits",
        help="Format of output file",
        config=True,
    )

    aliases = {
        ("i", "input"): "EventSource.input_url",
        ("f", "format"): "DumpInstrumentTool.format",
        ("o", "outdir"): "DumpInstrumentTool.outdir",
    }

    classes = [EventSource]

    def setup(self):
        with EventSource(parent=self) as source:
            self.infile = source.input_url
            self.subarray = source.subarray

    def start(self):
        if self.outdir is None:
            self.outdir = pathlib.Path(os.getcwd())

        self.outdir.mkdir(exist_ok=True, parents=True)

        if self.format == "hdf5":
            self.subarray.to_hdf(self.outdir / "subarray.h5")
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
            ext, args = self._get_file_format_info(self.format)

            self.log.debug("Writing camera %s", camera)
            geom = camera.geometry
            readout = camera.readout

            geom_table = geom.to_table()
            geom_table.meta["SOURCE"] = str(self.infile)
            geom_filename = self.outdir / f"{camera.name}.camgeom.{ext}"

            readout_table = readout.to_table()
            readout_table.meta["SOURCE"] = str(self.infile)
            readout_filename = self.outdir / f"{camera.name}.camreadout.{ext}"

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
        tab = sub.to_table(kind="subarray")
        tab.meta["SOURCE"] = str(self.infile)
        filename = self.outdir / f"{sub.name}.subarray.{ext}"
        try:
            tab.write(filename, **args)
            Provenance().add_output_file(filename, "SubarrayDescription")
        except OSError as err:
            self.log.exception(
                "couldn't write subarray description '%s' because: %s", filename, err
            )


def main():
    tool = DumpInstrumentTool()
    tool.run()


if __name__ == "__main__":
    main()
