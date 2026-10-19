"""
Display information about ctapipe output files (DL1 or DL2)
"""

from pathlib import Path

import ctao_datamodel as dm
import tables
import yaml
from astropy.table import Table

from ctapipe.io.metadata import read_ctao_metadata
from ctapipe.tools.utils import get_parser


def _read_file_info(filename, flat):
    """Read and format product metadata for one input file."""
    if not Path(filename).is_file():
        return "not a file", None

    if not tables.is_hdf5_file(filename):
        return "unknown file type", None

    try:
        with tables.open_file(filename, mode="r") as infile:
            product = read_ctao_metadata(infile)
            flat_metadata = dm.flatten_model_instance(product)
    except (tables.exceptions.HDF5ExtError, ValueError) as err:
        return f"ERROR {err}", None

    display_metadata = (
        flat_metadata
        if flat
        else product.model_dump(
            mode="json",
            exclude_none=True,
        )
    )
    table_row = {**flat_metadata, "PATH": filename}
    return display_metadata, table_row


def _write_index_table(files, output_path, table_format):
    """Write accumulated flattened metadata to an index table."""
    if output_path.endswith((".fits", ".fits.gz")):
        files = [
            {
                key: value.encode("utf-8") if isinstance(value, str) else value
                for key, value in row.items()
            }
            for row in files
        ]

    table = Table(files)
    table.write(output_path, format=table_format, overwrite=True)


def fileinfo(args):
    """
    Display information about ctapipe output files (DL1 or DL2 in HDF5 format).
    Optionally create an index table from all headers
    """

    files = []  # accumulated info for table output

    for filename in args.files:
        info, table_row = _read_file_info(filename, args.flat)
        print(yaml.dump({filename: info}, indent=4, sort_keys=False))

        if args.output_table and table_row is not None:
            files.append(table_row)

    if args.output_table:
        _write_index_table(files, args.output_table, args.table_format)


def main():
    """display info"""
    parser = get_parser(fileinfo)
    parser.add_argument(
        "files",
        metavar="FILENAME",
        type=str,
        nargs="+",
        help="filenames of files in ctapipe format",
    )
    parser.add_argument(
        "-o", "--output-table", help="generate output file in tabular format"
    )
    parser.add_argument(
        "-T",
        "--table-format",
        help="table format of output-table if not automatically guessed from filename",
    )
    parser.add_argument(
        "-f", "--flat", action="store_true", help="show flat header hierarchy"
    )
    args = parser.parse_args()

    fileinfo(args)


if __name__ == "__main__":
    main()
