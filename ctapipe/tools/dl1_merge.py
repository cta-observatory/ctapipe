"""
Merge DL1-files from stage1-process tool
"""
import pathlib
import sys
import tables
import os
import numpy as np

from tqdm.autonotebook import tqdm

from ctapipe.core import Tool
from ctapipe.instrument import SubarrayDescription
from ..core.traits import Bool, Path

import warnings

warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

table_path_list_base = ['/simulation/event/subarray/shower',
                        '/simulation/service/shower_distribution',
                        '/configuration/simulation/run',
                        '/dl1/monitoring/subarray/pointing',
                        '/dl1/event/telescope/trigger',
                        '/dl1/event/subarray/trigger']
group_path_list_images = ['/simulation/event/telescope/images',
                          '/dl1/event/telescope/images']
group_path_list_parameters = ['/simulation/event/telescope/parameters',
                              '/dl1/event/telescope/parameters']
group_path_pointings = '/dl1/monitoring/telescope/pointing'
table_path_image_statistics = '/dl1/service/image_statistics'


class DL1MergeTool(Tool):
    name = "ctapipe-merge-dl1"
    description = "Merges DL1-files from the stage1-process tool"
    examples = """
    To merge DL1-files created by the stage1-process-tool from a specific directory:
    > ctapipe-merge-dl1 --input=/path/to/directory --ouput=/path/to/output_file.h5 --progress
    """
    input_dir = Path(help="input dl1-directory",
                     exists=None, directory_ok=True, file_ok=False).tag(config=True)
    output_path = Path(help="Merged-DL1 output filename").tag(config=True)
    skip_images = Bool(help="Skip DL1/Event/Image data in output",
                       default_value=False).tag(config=True)
    skip_parameters = Bool(help="Skip image parameters",
                           default_value=False).tag(config=True)
    overwrite = Bool(help="overwrite output file if it exists").tag(config=True)
    progress_bar = Bool(help="show progress bar during processing").tag(config=True)

    aliases = {'input': 'DL1MergeTool.input_dir',
               'output': 'DL1MergeTool.output_path'}

    flags = {
        "skip-images": (
            {"DL1MergeTool": {"skip_images": True}},
            "store DL1/Event/Telescope images in output",
        ),
        "skip-parameters": (
            {"DL1MergeTool": {"skip_parameters": True}},
            "store DL1/Event/Telescope parameters in output",
        ),
        "overwrite": (
            {"DL1MergeTool": {"overwrite": True}},
            "Overwrite output file if it exists",
        ),
        "progress": (
            {"DL1MergeTool": {"progress_bar": True}},
            "show a progress bar during event processing",
        )
    }

    def setup(self):
        # prepare output path:
        self.output_path = self.output_path.expanduser()
        if self.output_path.exists():
            if self.overwrite:
                self.log.warning(f"Overwriting {self.output_path}")
                self.output_path.unlink()
            else:
                self.log.critical(f"Output file {self.output_path} exists, "
                                   "use `--overwrite` to overwrite")
                sys.exit(1)

        if self.skip_parameters is True and self.skip_images is True:
            self.log.warning('Skip-parameters and skip-images are both'
                             'set to True')

        self.input_path_list = sorted(self.input_dir.iterdir())

        # create output file with subarray from first file
        self.first_subarray = SubarrayDescription.from_hdf(self.input_path_list[0])
        self.first_subarray.to_hdf(self.output_path)
        self.output_file = tables.open_file(self.output_path, mode='a')

    def add_image_statistics(self, file):
        # Creates table for image statistics and adds the entries together.
        # This does not append rows to the existing table
        if table_path_image_statistics not in self.output_file:
            head, tail = os.path.split(os.path.split(table_path_image_statistics)[0])
            target_group = self.output_file.create_group(head, tail, createparents=True)
            file.copy_node(table_path_image_statistics, newparent=target_group)
            file.copy_node('/dl1/service/image_statistics.__table_column_meta__',
                           newparent=target_group)
        elif table_path_image_statistics in file:
            for row in range(3):
                (self.output_file.root[table_path_image_statistics].
                 cols.counts[row]) = np.add(
                      self.output_file.root[table_path_image_statistics].cols.counts[row],
                      file.root[table_path_image_statistics].cols.counts[row])
                (self.output_file.root[table_path_image_statistics].
                 cols.cumulative_counts[row]) = np.add(
                     (self.output_file.root[table_path_image_statistics].cols.
                      cumulative_counts[row]),
                      file.root[table_path_image_statistics].cols.cumulative_counts[row])

    def merge_tables(self, file):
        # 1) Create groups and copy tables if not in output_file
        # 2) Append to table if it already exists in output_file

        # For single tables in a group
        for table_path_base in table_path_list_base:
            if table_path_base in file and table_path_base in self.output_file:
                output_node = self.output_file.get_node(table_path_base)
                output_node.append(file.root[table_path_base][:])
            else:
                head, tail = os.path.split(os.path.split(table_path_base)[0])
                target_group = self.output_file.create_group(head, tail, createparents=True)
                file.copy_node(table_path_base, newparent=target_group)

        # For telescope pointing tables
        for table in file.root[group_path_pointings]:
            if group_path_pointings not in self.output_file:
                head, tail = os.path.split(group_path_pointings)
                self.output_file.create_group(head, tail, createparents=True)
            if (group_path_pointings + '/' + table.name) in self.output_file:
                output_node = self.output_file.get_node(group_path_pointings
                                                        + '/' + table.name)
                output_node.append(file.root[group_path_pointings + '/' + table.name][:])
            else:
                target_group = self.output_file.root[group_path_pointings]
                file.copy_node(table, newparent=target_group)

        # For telescope images
        if self.skip_images is False:
            for group_image in group_path_list_images:
                if group_image not in self.output_file:
                    head, tail = os.path.split(group_image)
                    self.output_file.create_group(head, tail, createparents=True)
                for table in file.root[group_image]:
                    if (group_image + '/' + table.name) in self.output_file:
                        output_node = self.output_file.get_node(group_image
                                                                + '/' + table.name)
                        output_node.append(file.root[group_image + '/' + table.name][:])
                    else:
                        target_group = self.output_file.root[group_image]
                        file.copy_node(table, newparent=target_group)

        # For telescope parameters
        if self.skip_parameters is False:
            for group_parameter in group_path_list_parameters:
                if group_parameter not in self.output_file:
                    head, tail = os.path.split(group_parameter)
                    self.output_file.create_group(head, tail, createparents=True)
                for table in file.root[group_parameter]:
                    if (group_parameter + '/' + table.name) in self.output_file:
                        output_node = self.output_file.get_node(group_parameter
                                                                + '/' + table.name)
                        output_node.append(file.root[group_parameter + '/' + table.name][:])
                    else:
                        target_group = self.output_file.root[group_parameter]
                        file.copy_node(table, newparent=target_group)

    def start(self):
        merged_files_counter = 0
        len_input_path_list = len(self.input_path_list)

        for i, current_file in tqdm(
            enumerate(self.input_path_list),
            desc="Merging",
            total=(len_input_path_list),
            unit="Files",
            disable=not self.progress_bar
        ):
            # Check if Subarray from current file matches to first one
            current_subarray = SubarrayDescription.from_hdf(current_file)
            if self.first_subarray != current_subarray:
                self.log.warning(f'Subarray does not match for {current_file}. Skip File')
                continue

            current_file_opened = tables.open_file(current_file, mode='r')
            self.add_image_statistics(current_file_opened)
            self.merge_tables(current_file_opened)
            current_file_opened.close()

            merged_files_counter += 1

        self.log.info(f"{merged_files_counter} out of {len_input_path_list} Files "
                       "had been merged!")

    def finish(self):
        pass


def main():
    tool = DL1MergeTool()
    tool.run()


if __name__ == "main":
   main()
