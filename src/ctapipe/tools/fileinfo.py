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


def fileinfo(args):
    """
    Display information about ctapipe output files (DL1 or DL2 in HDF5 format).
    Optionally create an index table from all headers
    """

    files = []  # accumulated info for table output

    for filename in args.files:
        info = {}

        # prevent failure if a non-file is given (e.g. a directory)
        if Path(filename).is_file() is False:
            info[filename] = "not a file"

        elif tables.is_hdf5_file(filename) is not True:
            info[filename] = "unknown file type"
        else:
            try:
                with tables.open_file(filename, mode="r") as infile:
                    product = read_ctao_metadata(infile)
                    flat_metadata = dm.flatten_model_instance(product)

                    if args.flat:
                        info[filename] = flat_metadata
                    else:
                        info[filename] = product.model_dump(
                            mode="json",
                            exclude_none=True,
                        )

                    if args.output_table:
                        table_row = flat_metadata.copy()
                        table_row["PATH"] = filename
                        files.append(table_row)

            except (tables.exceptions.HDF5ExtError, ValueError) as err:
                info[filename] = f"ERROR {err}"

        print(yaml.dump(info, indent=4))

    if args.output_table:
        if args.output_table.endswith(".fits") or args.output_table.endswith(
            ".fits.gz"
        ):
            files = [
                {
                    k: v.encode("utf-8") if isinstance(v, str) else v
                    for k, v in info.items()
                }
                for info in files
            ]

        table = Table(files)
        table.write(args.output_table, format=args.table_format, overwrite=True)


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
