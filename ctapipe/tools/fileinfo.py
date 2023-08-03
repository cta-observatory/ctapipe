"""
Display information about ctapipe output files (DL1 or DL2)
"""

from pathlib import Path

import tables
import yaml
from astropy.table import Table

from ..core import Tool, traits
from ..core.provenance import Provenance


def unflatten(dictionary, separator=" "):
    """turn flattened dict keys into nested"""
    hierarch_dict = dict()
    for key, value in dictionary.items():
        parts = key.split(separator)
        tmp_dict = hierarch_dict
        for part in parts[:-1]:
            if part not in tmp_dict:
                tmp_dict[part] = dict()
            tmp_dict = tmp_dict[part]
        tmp_dict[parts[-1]] = value
    return hierarch_dict


class FileInfoTool(Tool):
    """Extract metadata and other information from ctapipe output files"""

    name = "ctapipe-fileinfo"
    description = __doc__
    examples = """To get YAML output of all metadata in HDF5 files in the
    current directory

    > ctapipe fileinfo *.h5

    Generate an index table of all metadata: Note that you can
    use any table format allowed by astropy.table. However, formats
    with metadata like fits or ecsv are recommended.

    > ctapipe fileinfo --output-table index.fits *.h5"""

    input_files = traits.List(
        traits.Path(exists=True, directory_ok=False),
        default_value=[],
        help=(
            "Input ctapipe HDF5 files. These can also be "
            "specified as positional command-line arguments."
        ),
    ).tag(config=True)

    flat = traits.Bool(False, help="Flatten metadata hierarchy").tag(config=True)
    output_table = traits.Path(
        None,
        exists=False,
        directory_ok=False,
        file_ok=True,
        allow_none=True,
        help=(
            "Filename of output index table with all file information. "
            "This can be in any format supported by astropy.table. The output format is "
            "guessed from the filename, or you can specify it explicity using the "
            "table_format option. E.g: 'index.ecsv', 'index.fits', 'index.html'. "
        ),
    ).tag(config=True)

    table_format = traits.Unicode(
        None,
        allow_none=True,
        help="Table format for output-table if not automatically guessed from the filename",
    ).tag(config=True)

    aliases = {
        ("i", "input-files"): "FileInfoTool.input_files",
        ("T", "table-format"): "FileInfoTool.table_format",
        ("o", "output-table"): "FileInfoTool.output_table",
    }

    flags = {
        "flat": ({"FileInfoTool": {"flat": True}}, "Flatten metadata hierarchy"),
    }

    def setup(self):
        # Get input Files from positional arguments
        positional_input_files = self.__class__.input_files.validate_elements(
            self, self.extra_args
        )
        self.input_files.extend(positional_input_files)

    def start(self):
        """
        Display information about ctapipe output files (DL1 or DL2 in HDF5 format).
        Optionally create an index table from all headers
        """

        files = []  # accumulated info for table output

        for filename in self.input_files:
            info = {}
            filename = str(filename)

            # prevent failure if a non-file is given (e.g. a directory)
            if Path(filename).is_file() is False:
                info[filename] = "not a file"

            elif tables.is_hdf5_file(filename) is not True:
                info[filename] = "unknown file type"
            else:
                try:
                    with tables.open_file(filename, mode="r") as infile:
                        Provenance().add_input_file(
                            filename, role="ctapipe-fileinfo input file"
                        )
                        # pylint: disable=W0212,E1101
                        attrs = {
                            name: str(infile.root._v_attrs[name])
                            for name in infile.root._v_attrs._f_list()
                        }
                        if self.flat:
                            info[filename] = attrs.copy()
                        else:
                            info[filename] = unflatten(attrs)

                        if self.output_table:
                            attrs["PATH"] = filename
                            files.append(attrs)

                except tables.exceptions.HDF5ExtError as err:
                    info[filename] = f"ERROR {err}"

            print(yaml.dump(info, indent=4))

        if self.output_table:
            if ".fits" in self.output_table.suffixes:
                # need to add proper string encoding for FITS, otherwise the
                # conversion fails (libHDF5 gives back raw bytes, not python strings)
                files = [
                    {k: v.encode("utf-8") for k, v in info.items()} for info in files
                ]

            table = Table(files)
            table.write(
                self.output_table, format=self.table_format, overwrite=self.overwrite
            )
            Provenance().add_output_file(
                self.output_table, role="ctapipe-fileinfo table"
            )

    def finish(self):
        pass


def main():
    """display info"""
    tool = FileInfoTool()
    tool.run()


if __name__ == "__main__":
    main()
