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

allowlist_base = ['/dl1/monitoring/subarray/pointing',
                  '/dl1/monitoring/telescope/pointing',
                  '/dl1/service/image_statistics',
                  '/dl1/service/image_statistics.__table_column_meta__',
                  '/dl1/event/subarray/trigger',
                  '/dl1/event/telescope/trigger',
                  '/dl1/event/telescope/parameters',
                  '/dl1/event/telescope/images',
                  '/configuration/simulation/run',
                  '/simulation/event/subarray/shower',
                  '/simulation/event/telescope/parameters',
                  '/simulation/event/telescope/images',
                  '/simulation/service/shower_distribution']
allowlist_simu = ['/simulation/event/subarray/shower',
                  '/simulation/event/telescope/parameters',
                  '/simulation/event/telescope/images',
                  '/simulation/service/shower_distribution']
allowlist_service = ['/dl1/service/image_statistics',
                     '/dl1/service/image_statistics.__table_column_meta__']
allowlist_unessential = ['/simulation/service/shower_distribution',
                         '/simulation/event/telescope/images',
                         '/simulation/event/telescope/parameters']
allowlist_tels = ['/dl1/monitoring/telescope/pointing',
                  '/dl1/event/telescope/parameters',
                  '/dl1/event/telescope/images',
                  '/simulation/event/telescope/parameters',
                  '/simulation/event/telescope/images']
blocklist_images = ['/simulation/event/telescope/images',
                    '/dl1/event/telescope/images']
blocklist_parameters = ['/simulation/event/telescope/parameters',
                        '/dl1/event/telescope/parameters']


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
    input_dir = traits.Path(help="input dl1-directory",
                            exists=None, directory_ok=True,
                            file_ok=False).tag(config=True)
    input_files = List(default_value=[],
                       help="input dl1-files").tag(config=True)
    output_path = traits.Path(help="Merged-DL1 output filename").tag(config=True)
    skip_images = traits.Bool(help="Skip DL1/Event/Image data in output",
                              default_value=False).tag(config=True)
    skip_simu_images = traits.Bool(help="Skip DL1/Event/Image data in output",
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

    aliases = {'input-dir': 'MergeTool.input_dir',
               'output': 'MergeTool.output_path',
               'pattern' : 'MergeTool.file_pattern'}

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
        if self.input_dir is not None:
            self.input_files.extend(sorted(self.input_dir.glob(self.file_pattern)))
        if not self.input_files:
            self.input_files = sorted(Path('.').glob(self.file_pattern))

        # create output file with subarray from first file
        self.first_subarray = SubarrayDescription.from_hdf(self.input_files[0])
        self.first_subarray.to_hdf(self.output_path)
        self.output_file = tables.open_file(self.output_path, mode='a')

    def check_file(self, file):
        # Check that the file is not broken or any node is missing
        file_path = file.root._v_file.filename
        current_subarray = SubarrayDescription.from_hdf(file_path)
        # Check subarrays
        if self.first_subarray != current_subarray:
            self.log.warning(f"Subarray does not match for {file_path}. "
                             "Skip File.")
            self.skip_file = True

        # Gives warning if tables for listed nodes in 'allowlist_unessential'
        # are missing but continues merging the rest of the file. If any
        # other node from 'allowlist_base' is missing, except the given combinations
        # of traits and blocklist, the whole file will be skipped.
        for node in allowlist_base:
            if self.is_simu is False and node in allowlist_simu:
                continue
            if self.skip_simu_images is True and node is blocklist_images[0]:
                continue
            if self.skip_images is True and node in blocklist_images:
                continue
            if self.skip_parameters is True and node in blocklist_parameters:
                continue
            if node in allowlist_unessential and node not in file:
                self.log.warning(f"{node} is not in {file_path}. Continue with "
                                 "merging file. This table will be incomplete "
                                 "or empty")
                continue
            if node not in file:
                self.log.warning(f"{node} is not in {file_path}. Skip file")
                self.skip_file = True

    def add_image_statistics(self, file):
        # Creates table for image statistics and adds the entries together.
        # This does not append rows to the existing table
        image_statistics_path = allowlist_service[0]
        if image_statistics_path in self.output_file:
            table_out = self.output_file.root[image_statistics_path]
            table_in = file.root[image_statistics_path]
            for row in range(len(table_in)):
                table_out.cols.counts[row] = np.add(table_out.cols.counts[row],
                                                    table_in.cols.counts[row])
                table_out.cols.cumulative_counts[row] = np.add(
                    table_out.cols.cumulative_counts[row],
                    table_in .cols.cumulative_counts[row])

        elif '/dl1/service' not in self.output_file:
            target_group = self.output_file.create_group('/dl1',
                                                         'service', createparents=True)
            for node in allowlist_service:
                file.copy_node(node, newparent=target_group)

    def merge_tables(self, file):
        # Loop over all nodes listed in allowlist_base. Appends table to output_file
        # if it already exists, otherwise creates group and copies node. If skip_images
        # or skip_simu_images or skip_parameters flag is True, related groups
        # will be skipped.
        for node in allowlist_base:
            if node in file:
                if self.skip_images is True and node in blocklist_images:
                    continue
                elif self.skip_simu_images is True and node in blocklist_images[0]:
                    continue
                if self.skip_parameters is True and node in blocklist_parameters:
                    continue
                if node in allowlist_service:
                    continue
                if node in allowlist_tels:
                    if node not in self.output_file:
                        head, tail = os.path.split(node)
                        self.output_file.create_group(head, tail, createparents=True)
                    for tel in file.root[node]:
                        if (node + '/' + tel.name) in self.output_file:
                            output_node = self.output_file.get_node(node + '/' + tel.name)
                            output_node.append(file.root[node + '/' + tel.name][:])
                        else:
                            target_group = self.output_file.root[node]
                            file.copy_node(tel, newparent=target_group)
                    continue
                if node in self.output_file:
                    output_node = self.output_file.get_node(node)
                    output_node.append(file.root[node][:])
                else:
                    group_path = os.path.split(node)[0]
                    if group_path not in self.output_file:
                        head, tail = os.path.split(group_path)
                        self.output_file.create_group(head, tail, createparents=True)
                    target_group = self.output_file.root[group_path]
                    file.copy_node(node, newparent=target_group)

    def start(self):
        merged_files_counter = 0
        self.is_simu = False

        for i, current_file in enumerate(tqdm(
            self.input_files,
            desc="Merging",
            unit="Files",
            disable=not self.progress_bar
        )):

            self.skip_file = False
            with tables.open_file(current_file, mode='r') as file:
                if i == 0:
                    # Check if first file is simulation
                    if '/simulation' in file:
                        self.is_simu = True
                        self.log.info('Merging simulation-files')
                self.check_file(file)
                if self.skip_file is True:
                    continue
                else:
                    self.merge_tables(file)
                    self.add_image_statistics(file)

            merged_files_counter += 1

        self.log.info(f"{merged_files_counter} out of {len(self.input_files)} files "
                      "has been merged!")

    def finish(self):
        pass


def main():
    tool = MergeTool()
    tool.run()


if __name__ == "main":
    main()
