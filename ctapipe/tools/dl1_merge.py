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

warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

PROV = Provenance()

blacklist_path = ['/configuration/instrument/subarray',
                  '/configuration/instrument/telescope',
                  '/configuration/instrument/telescope/camera',
                  '/dl1/service']
blacklist_images = ['/simulation/event/telescope/images',
                    '/dl1/event/telescope/images']
blacklist_parameters = ['/simulation/event/telescope/parameters',
                        '/dl1/event/telescope/parameters']
service_group = '/dl1/service'


class MergeTool(Tool):
    name = "ctapipe-merge"
    description = "Merges DL1-files from the stage1-process tool"
    examples = """
    To merge DL1-files created by the stage1-process-tool from current directory:

    > ctapipe-merge file1.h5 file2.h5 file3.h5 --ouput=/path/output_file.h5 --progress

    If you want a specific file pattern as input files use --patter='pattern.*.dl1.h5', e.g.:

    > ctapip-merge --output=/path/output_file.h5 --progress --pattern='pattern.*.dl1.h5'

    If neither any input files nor any pattern is given, all files from the current directory
    with the default pattern '*.h5' are taken.
    """
    input_dir = traits.Path(help="input dl1-directory",
                            exists=None, directory_ok=True,
                            file_ok=False).tag(config=True)
    input_files = List(default_value=[],
                       help="input dl1-files").tag(config=True)
    output_path = traits.Path(help="Merged-DL1 output filename").tag(config=True)
    skip_images = traits.Bool(help="Skip DL1/Event/Image data in output",
                              default_value=False).tag(config=True)
    skip_parameters = traits.Bool(help="Skip image parameters",
                                  default_value=False).tag(config=True)
    overwrite = traits.Bool(help="overwrite output file if it exists").tag(config=True)
    progress_bar = traits.Bool(help="show progress bar during "
                                    "processing").tag(config=True)
    file_pattern = traits.Unicode(default_value='*.h5',
                                  help="Give a specific file pattern for the"
                                       "input files").tag(config=True)

    parser = ArgumentParser()
    parser.add_argument('input_files', nargs='*', type=Path)

    aliases = {'input_dir': 'Merge.Tool.input_dir',
               'output': 'MergeTool.output_path',
               'pattern' : 'MergeTool.file_pattern'}

    flags = {
        "skip-images": (
            {"MergeTool": {"skip_images": True}},
            "Skip DL1/Event/Telescope images in output",
        ),
        "skip-parameters": (
            {"MergeTool": {"skip_parameters": True}},
            "Skip DL1/Event/Telescope parameters in output",
        ),
        "overwrite": (
            {"MergeTool": {"overwrite": True}},
            "Overwrite output file if it exists",
        ),
        "progress": (
            {"MergeTool": {"progress_bar": True}},
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

        PROV.add_output_file(str(self.output_path))

        if self.skip_parameters is True and self.skip_images is True:
            self.log.warning('Skip-parameters and skip-images are both'
                             'set to True')

        # Get input Files
        args = self.parser.parse_args(self.extra_args)
        self.input_files = args.input_files
        if not self.input_files:
            self.input_files = sorted(Path('.').glob(self.file_pattern))
        if self.input_dir is not None:
            self.input_files.extend(sorted(self.input_dir.glob(self.file_pattern)))

        # create output file with subarray from first file
        self.first_subarray = SubarrayDescription.from_hdf(self.input_files[0])
        self.first_subarray.to_hdf(self.output_path)
        self.output_file = tables.open_file(self.output_path, mode='a')

    def check_file(self, file):
        # Check that the file is not broken
        return

    def add_image_statistics(self, file):
        # Creates table for image statistics and adds the entries together.
        # This does not append rows to the existing table
        if service_group in file:
            image_statistics_path = service_group + '/image_statistics'
            if image_statistics_path in file and \
               image_statistics_path in self.output_file:
                table_out = self.output_file.root[image_statistics_path]
                table_in = file.root[image_statistics_path]
                for row in range(len(table_in)):
                    table_out.cols.counts[row] = np.add(table_out.cols.counts[row],
                                                        table_in.cols.counts[row])
                    table_out.cols.cumulative_counts[row] = np.add(
                        table_out.cols.cumulative_counts[row],
                        table_in .cols.cumulative_counts[row])

            elif service_group not in self.output_file:
                head, tail = os.path.split(service_group)
                target_group = self.output_file.create_group(head,
                                                             tail, createparents=True)
                file.copy_node(image_statistics_path, newparent=target_group)
                file.copy_node(image_statistics_path + '.__table_column_meta__',
                               newparent=target_group)

    def merge_tables(self, file):
        # Loop over all groups and and tables of that group. Appends table to output_file
        # if it already exists, otherwise creates group and copies node. If skip_images
        # or skip_parameters flag is True, related group will be skipped

        for group in file.walk_groups(where='/'):
            group_path = file.get_node_attr(group, '_v__nodepath')
            if group_path in blacklist_path:
                continue
            if self.skip_images is True and group_path in blacklist_images:
                continue
            if self.skip_parameters is True and group_path in blacklist_parameters:
                continue
            for node in file.iter_nodes(group, classname='Table'):
                if (group_path + '/' + node.name) in self.output_file:
                    output_node = self.output_file.get_node(group_path + '/' + node.name)
                    output_node.append(file.root[group_path + '/' + node.name][:])
                elif (group_path + '/' + node.name) not in self.output_file:
                    if group_path not in self.output_file:
                        head, tail = os.path.split(group_path)
                        self.output_file.create_group(head, tail, createparents=True)
                    target_group = self.output_file.root[group_path]
                    file.copy_node(node, newparent=target_group)

    def start(self):
        merged_files_counter = 0
        len_input_files = len(self.input_files)

        for current_file in tqdm(
            self.input_files,
            desc="Merging",
            total=(len_input_files),
            unit="Files",
            disable=not self.progress_bar
        ):
            # Check if Subarray from current file matches to first one
            current_subarray = SubarrayDescription.from_hdf(current_file)
            if self.first_subarray != current_subarray:
                self.log.warning(f'Subarray does not match for {current_file}. Skip File')
                continue

            with tables.open_file(current_file, mode='r') as file:
                self.merge_tables(file)
                self.add_image_statistics(file)

            merged_files_counter += 1

        self.log.info(f"{merged_files_counter} out of {len_input_files} Files "
                      "has been merged!")

    def finish(self):
        pass


def main():
    tool = MergeTool()
    tool.run()


if __name__ == "main":
    main()
