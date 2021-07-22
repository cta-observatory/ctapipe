"""
Display information about ctapipe output files (DL1 or DL2)
"""

from pathlib import Path
import tables
import yaml
from astropy.table import Table
from ctapipe.tools.utils import get_parser


def unflatten(dictionary, separator=" "):
    """ turn flattened dict keys into nested """
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


def fileinfo(args):
    """
    Display information about ctapipe output files (DL1 or DL2 in HDF5 format).
    Optionally create an index table from all headers
    """

    info_total = {}  # accumulated info for table output

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
                    # pylint: disable=W0212,E1101
                    attrs = {
                        name: str(infile.root._v_attrs[name])
                        for name in infile.root._v_attrs._f_list()
                    }
                    if args.flat:
                        info[filename] = attrs
                    else:
                        info[filename] = unflatten(attrs)

                    if args.output_table:
                        info_total[filename] = attrs
            except tables.exceptions.HDF5ExtError as err:
                info[filename] = f"ERROR {err}"

        print(yaml.dump(info, indent=4))

    if args.output_table:
        # use pandas' ability to convert a dict of flat values to a table
        import pandas as pd  #  pylint: disable=C0415

        dataframe = pd.DataFrame(info_total)
        Table.from_pandas(dataframe.T, index=True).write(
            args.output_table, overwrite=True
        )


def main():
    """ display info """
    parser = get_parser(fileinfo)
    parser.add_argument(
        "files",
        metavar="FILENAME",
        type=str,
        nargs="+",
        help="filenames of files in ctapipe format",
    )
    parser.add_argument("--output-table", help="generate output in tabular format")
    parser.add_argument(
        "--flat", action="store_true", help="show flat header hierarchy"
    )
    args = parser.parse_args()

    fileinfo(args)


if __name__ == "__main__":
    main()
