"""
Merge DL1-files from stage1-process tool
"""
from pathlib import Path
import sys
import tables
import os
from argparse import ArgumentParser
import numpy as np

from tqdm import tqdm

from ..core import Provenance
from ctapipe.core import Tool, traits
from traitlets import List
from ctapipe.instrument import SubarrayDescription

import warnings

warnings.filterwarnings("ignore", category=tables.NaturalNameWarning)

PROV = Provenance()

all_nodes = [
    "/dl1/monitoring/subarray/pointing",
    "/dl1/monitoring/telescope/pointing",
    "/dl1/service/image_statistics",
    "/dl1/service/image_statistics.__table_column_meta__",
    "/dl1/event/subarray/trigger",
    "/dl1/event/telescope/trigger",
    "/dl1/event/telescope/parameters",
    "/dl1/event/telescope/images",
    "/configuration/simulation/run",
    "/simulation/event/subarray/shower",
    "/simulation/event/telescope/parameters",
    "/simulation/event/telescope/images",
    "/simulation/service/shower_distribution",
]
optional_nodes = [
    "/simulation/service/shower_distribution",
    "/simulation/event/telescope/images",
    "/simulation/event/telescope/parameters",
    "/dl1/event/telescope/parameters",
    "/dl1/event/telescope/images",
    "/dl1/service/image_statistics",
    "/dl1/service/image_statistics.__table_column_meta__",
]
simu_nodes = [
    "/simulation/event/subarray/shower",
    "/simulation/event/telescope/parameters",
    "/simulation/event/telescope/images",
    "/simulation/service/shower_distribution",
    "/configuration/simulation/run",
]
service_nodes = [
    "/dl1/service/image_statistics",
    "/dl1/service/image_statistics.__table_column_meta__",
]
nodes_with_tels = [
    "/dl1/monitoring/telescope/pointing",
    "/dl1/event/telescope/parameters",
    "/dl1/event/telescope/images",
    "/simulation/event/telescope/parameters",
    "/simulation/event/telescope/images",
]


class MergeTool(Tool):
    name = "ctapipe-merge"
    description = "Merges DL1-files from the stage1-process tool"
    examples = """
    To merge DL1-files created by the stage1-process-tool from current directory:

    > ctapipe-merge file1.h5 file2.h5 file3.h5 --ouput=/path/output_file.h5 --progress

    If you want a specific file pattern as input files use --pattern='pattern.*.dl1.h5', e.g.:

    > ctapipe-merge --output=/path/output_file.h5 --progress --pattern='pattern.*.dl1.h5'

    If neither any input files nor any pattern is given, all files from the current directory
    with the default pattern '*.h5' are taken. For merging all files from a specific
    directory with a given pattern, use:

    > ctapipe-merge --input-dir=/input/dir/ --output=/path/output_file.h5 --progress
    --pattern='pattern.*.dl1.h5'
    """
    input_dir = traits.Path(
        help="Input dl1-directory", exists=True, directory_ok=True, file_ok=False
    ).tag(config=True)
    input_files = List(default_value=[], help="Input dl1-files").tag(config=True)
    output_path = traits.Path(
        help="Merged-DL1 output filename",
        directory_ok=False,
    ).tag(config=True)
    skip_images = traits.Bool(
        help="Skip DL1/Event/Telescope and Simulation/Event/Telescope images in output",
        default_value=False,
    ).tag(config=True)
    skip_simu_images = traits.Bool(
        help="Skip Simulation/Event/Telescope images in output",
        default_value=False,
    ).tag(config=True)
    skip_parameters = traits.Bool(
        help="Skip DL1/Event/Telescope and Simulation/Event/Telescope parameters"
        "in output",
        default_value=False,
    ).tag(config=True)
    skip_broken_files = traits.Bool(
        help="Skip broken files instead of giving error",
        default_value=False,
    ).tag(config=True)
    overwrite = traits.Bool(help="Overwrite output file if it exists").tag(config=True)
    progress_bar = traits.Bool(help="Show progress bar during " "processing").tag(
        config=True
    )
    file_pattern = traits.Unicode(
        default_value="*.h5", help="Give a specific file pattern for the" "input files"
    ).tag(config=True)

    parser = ArgumentParser()
    parser.add_argument("input_files", nargs="*", type=Path)

    aliases = {
        "i": "MergeTool.input_dir",
        "o": "MergeTool.output_path",
        "p": "MergeTool.file_pattern",
    }

    flags = {
        "skip-images": (
            {"MergeTool": {"skip_images": True}},
            "Skip DL1/Event/Telescope and Simulation/Event/Telescope images in output",
        ),
        "skip-simu-images": (
            {"MergeTool": {"skip_simu_images": True}},
            "Skip Simulation/Event/Telescope images in output",
        ),
        "skip-parameters": (
            {"MergeTool": {"skip_parameters": True}},
            "Skip DL1/Event/Telescope and Simulation/Event/Telescope parameters in output",
        ),
        "skip-broken-files": (
            {"MergeTool": {"skip_broken_files": True}},
            "Skip broken files instead of giving error",
        ),
        "overwrite": (
            {"MergeTool": {"overwrite": True}},
            "Overwrite output file if it exists",
        ),
        "progress": (
            {"MergeTool": {"progress_bar": True}},
            "Show a progress bar during event processing",
        ),
    }

    def setup(self):
        # prepare output path:
        self.output_path = self.output_path.expanduser()
        if self.output_path.exists():
            if self.overwrite:
                self.log.warning(f"Overwriting {self.output_path}")
                self.output_path.unlink()
            else:
                self.log.critical(
                    f"Output file {self.output_path} exists, "
                    "use `--overwrite` to overwrite"
                )
                sys.exit(1)

        PROV.add_output_file(str(self.output_path))

        if self.skip_parameters is True and self.skip_images is True:
            self.log.warning("Skip-parameters and skip-images are both set to True")
            sys.exit(1)

        # Get input Files
        args = self.parser.parse_args(self.extra_args)
        self.input_files.extend(args.input_files)
        if self.input_dir is not None:
            self.input_files.extend(sorted(self.input_dir.glob(self.file_pattern)))

        if not self.input_files:
            self.log.critical(
                "No input files provided, either provide --input-dir "
                "or input files as positional arguments"
            )
            sys.exit(1)

        # create output file with subarray from first file
        self.first_subarray = SubarrayDescription.from_hdf(self.input_files[0])
        self.first_subarray.to_hdf(self.output_path)
        self.output_file = tables.open_file(self.output_path, mode="a")

        # setup required nodes
        self.usable_nodes = set(all_nodes)

        if self.skip_simu_images is True:
            simu_images = ["/simulation/event/telescope/images"]
            self.usable_nodes = self.usable_nodes - set(simu_images)

        if self.skip_images is True:
            images_path = [
                "/simulation/event/telescope/images",
                "/dl1/event/telescope/images",
            ]
            self.usable_nodes = self.usable_nodes - set(images_path)

        if self.skip_parameters is True:
            parameters_path = [
                "/simulation/event/telescope/parameters",
                "/dl1/event/telescope/parameters",
            ]
            self.usable_nodes = self.usable_nodes - set(parameters_path)

    def check_file(self, file):
        # Check that the file is not broken or any node is missing
        file_path = file.root._v_file.filename
        current_subarray = SubarrayDescription.from_hdf(file_path)

        # Check subarray
        if self.first_subarray != current_subarray:
            self.log.critical(f"Subarray does not match for {file_path}")
            self.broken_file = True

        # Gives warning if tables for listed nodes in 'optional_nodes'
        # are missing but continues merging the rest of the file. Gives error
        # if any other node from 'usable_nodes' is missing, except
        # 'skip_broken_files' is set to True, then skip files.
        for node in self.usable_nodes:
            if node in optional_nodes and node not in file:
                self.log.warning(
                    f"{node} is not in {file_path}. Continue with "
                    "merging file. This table will be incomplete "
                    "or empty"
                )
                continue

            if node not in file:
                self.log.critical(f"{node} is not in {file_path}.")
                self.broken_file = True

    def add_image_statistics(self, file):
        # Creates table for image statistics and adds the entries together.
        # This does not append rows to the existing table
        image_statistics_path = "/dl1/service/image_statistics"
        if image_statistics_path in self.output_file:
            table_out = self.output_file.root[image_statistics_path]
            table_in = file.root[image_statistics_path]
            for row in range(len(table_in)):
                table_out.cols.counts[row] = np.add(
                    table_out.cols.counts[row], table_in.cols.counts[row]
                )
                table_out.cols.cumulative_counts[row] = np.add(
                    table_out.cols.cumulative_counts[row],
                    table_in.cols.cumulative_counts[row],
                )

        elif "/dl1/service" not in self.output_file:
            target_group = self.output_file.create_group(
                "/dl1", "service", createparents=True
            )
            for node in set(service_nodes):
                file.copy_node(node, newparent=target_group)

    def merge_tables(self, file):
        # Loop over all nodes listed in usable_nodes. Appends table to output_file
        # if it already exists, otherwise creates group and copies node.
        for node in self.usable_nodes:
            if node in service_nodes:
                continue

            if node in file:
                if node in nodes_with_tels:
                    if node not in self.output_file:
                        self._create_group(node)

                    for tel in file.root[node]:
                        tel_node_path = node + "/" + tel.name
                        if tel_node_path in self.output_file:
                            output_node = self.output_file.get_node(tel_node_path)
                            input_node = file.root[tel_node_path]

                            # cast needed for some image parameters that are sometimes
                            # float32 and sometimes float64
                            output_node.append(input_node[:].astype(output_node.dtype))

                        else:
                            target_group = self.output_file.root[node]
                            file.copy_node(tel, newparent=target_group)

                    continue

                if node in self.output_file:
                    # this is needed to merge 0.8 files due to
                    # a column added erronouesly which prevents merging
                    # because it's variable length
                    # TODO: remove when we no longer want to support merging 0.8 files.
                    data = file.root[node][:]
                    if node == "/dl1/monitoring/subarray/pointing":
                        data = self.drop_column(data, "tels_with_trigger")

                    output_node = self.output_file.get_node(node)
                    output_node.append(data)

                else:
                    group_path, table_name = os.path.split(node)
                    if group_path not in self.output_file:
                        self._create_group(group_path)

                    target_group = self.output_file.root[group_path]

                    if node == "/dl1/monitoring/subarray/pointing":
                        h5_node = file.root[node]
                        data = self.drop_column(h5_node[:], "tels_with_trigger")
                        self.output_file.create_table(
                            group_path,
                            table_name,
                            filters=h5_node.filters,
                            obj=data,
                        )

                    else:
                        file.copy_node(node, newparent=target_group)

    def _create_group(self, node):
        head, tail = os.path.split(node)
        self.output_file.create_group(head, tail, createparents=True)

    @staticmethod
    def drop_column(array, column):
        from numpy.lib.recfunctions import repack_fields

        cols = list(array.dtype.names)
        if column in cols:
            cols.remove(column)

        return repack_fields(array[cols])

    def start(self):
        merged_files_counter = 0

        for i, current_file in enumerate(
            tqdm(
                self.input_files,
                desc="Merging",
                unit="Files",
                disable=not self.progress_bar,
            )
        ):

            self.broken_file = False
            with tables.open_file(current_file, mode="r") as file:
                if i == 0:
                    # Check if first file is simulation
                    if "/simulation" not in file.root:
                        self.usable_nodes = self.usable_nodes - set(simu_nodes)
                    else:
                        self.log.info("Merging simulation-files")

                self.check_file(file)
                if self.broken_file is True and self.skip_broken_files is True:
                    continue
                if self.broken_file is True and self.skip_broken_files is False:
                    self.log.critical("Broken file detected.")
                    sys.exit(1)

                self.merge_tables(file)
                self.add_image_statistics(file)

            PROV.add_input_file(str(current_file))
            merged_files_counter += 1

        self.log.info(
            f"{merged_files_counter} out of {len(self.input_files)} files "
            "has been merged!"
        )

    def finish(self):
        pass


def main():
    tool = MergeTool()
    tool.run()


if __name__ == "main":
    main()
