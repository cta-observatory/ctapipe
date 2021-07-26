"""
Merge DL1-files from stage1-process tool
"""
import sys
import os
from argparse import ArgumentParser
from pathlib import Path
from traitlets import List

import tables
import numpy as np
from tqdm.auto import tqdm

from ..io import metadata as meta, DL1EventSource
from ..io import HDF5TableWriter
from ..core import Provenance, Tool, traits
from ..core.traits import Bool, Set, Unicode, flag, CInt
from ..instrument import SubarrayDescription

import warnings

warnings.filterwarnings("ignore", category=tables.NaturalNameWarning)

PROV = Provenance()

VERSION_KEY = "CTA PRODUCT DATA MODEL VERSION"
IMAGE_STATISTICS_PATH = "/dl1/service/image_statistics"

all_nodes = {
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
}
optional_nodes = {
    "/simulation/service/shower_distribution",
    "/simulation/event/telescope/images",
    "/simulation/event/telescope/parameters",
    "/dl1/event/telescope/parameters",
    "/dl1/event/telescope/images",
    "/dl1/service/image_statistics",
    "/dl1/service/image_statistics.__table_column_meta__",
}
simu_nodes = {
    "/simulation/event/subarray/shower",
    "/simulation/event/telescope/parameters",
    "/simulation/event/telescope/images",
    "/simulation/service/shower_distribution",
    "/configuration/simulation/run",
}
service_nodes = {
    "/dl1/service/image_statistics",
    "/dl1/service/image_statistics.__table_column_meta__",
}
nodes_with_tels = {
    "/dl1/monitoring/telescope/pointing",
    "/dl1/event/telescope/parameters",
    "/dl1/event/telescope/images",
    "/simulation/event/telescope/parameters",
    "/simulation/event/telescope/images",
}
image_nodes = {"/simulation/event/telescope/images", "/dl1/event/telescope/images"}
parameter_nodes = {
    "/simulation/event/telescope/parameters",
    "/dl1/event/telescope/parameters",
}
simu_images = {"/simulation/event/telescope/images"}


class MergeTool(Tool):
    name = "ctapipe-merge"
    description = "Merges DL1-files created by the stage1-tool"
    examples = """
    To merge DL1-files created by the stage1-tool from current directory:

    > ctapipe-merge file1.h5 file2.h5 file3.h5 --output=/path/output_file.h5 --progress

    For merging files from a specific directory with a specific pattern, use:

    > ctapipe-merge --input-dir=/input/dir/ --output=/path/output_file.h5 --progress
    --pattern='*.dl1.h5'

    If no pattern is given, all .h5 files of the given directory will be taken as input.
    """
    input_dir = traits.Path(
        help="Input dl1-directory", exists=True, directory_ok=True, file_ok=False
    ).tag(config=True)
    input_files = List(default_value=[], help="Input dl1-files").tag(config=True)
    output_path = traits.Path(
        help="Merged-DL1 output filename", directory_ok=False
    ).tag(config=True)
    skip_images = Bool(
        help="Skip DL1/Event/Telescope and Simulation/Event/Telescope images in output",
        default_value=False,
    ).tag(config=True)
    skip_simu_images = Bool(
        help="Skip Simulation/Event/Telescope images in output", default_value=False
    ).tag(config=True)
    skip_parameters = Bool(
        help="Skip DL1/Event/Telescope and Simulation/Event/Telescope parameters"
        "in output",
        default_value=False,
    ).tag(config=True)
    skip_broken_files = Bool(
        help="Skip broken files instead of raising an error", default_value=False
    ).tag(config=True)
    overwrite = Bool(help="Overwrite output file if it exists").tag(config=True)
    progress_bar = Bool(help="Show progress bar during processing").tag(config=True)
    file_pattern = Unicode(
        default_value="*.h5", help="Give a specific file pattern for the input files"
    ).tag(config=True)
    allowed_tels = Set(
        trait=CInt(),
        default_value=None,
        allow_none=True,
        help=(
            "list of allowed tel_ids, others will be ignored. "
            "If None, all telescopes in the input stream "
            "will be included"
        ),
    ).tag(config=True)

    parser = ArgumentParser()
    parser.add_argument("input_files", nargs="*", type=Path)

    aliases = {
        ("i", "input-dir"): "MergeTool.input_dir",
        ("o", "output"): "MergeTool.output_path",
        ("p", "pattern"): "MergeTool.file_pattern",
        ("t", "allowed-tels"): "MergeTool.allowed_tels",
    }

    flags = {
        "f": ({"MergeTool": {"overwrite": True}}, "Overwrite output file if it exists"),
        **flag(
            "overwrite",
            "MergeTool.overwrite",
            "Overwrite output file if it exists",
            "Don't overwrite output file if it exists",
        ),
        "progress": (
            {"MergeTool": {"progress_bar": True}},
            "Show a progress bar for all given input files",
        ),
        **flag(
            "skip-images",
            "MergeTool.skip_images",
            "Skip DL1/Event/Telescope and Simulation/Event/Telescope images in output",
            "Don't skip DL1/Event/Telescope and Simulation/Event/Telescope images in output",
        ),
        **flag(
            "skip-simu-images",
            "MergeTool.skip_simu_images",
            "Skip Simulation/Event/Telescope images in output",
            "Don't skip Simulation/Event/Telescope images in output",
        ),
        **flag(
            "skip-parameters",
            "MergeTool.skip_parameters",
            "Skip DL1/Event/Telescope and Simulation/Event/Telescope parameters in output",
            "Don't skip DL1/Event/Telescope and Simulation/Event/Telescope parameters in output",
        ),
        **flag(
            "skip-broken-files",
            "MergeTool.skip_broken_files",
            "Skip broken files instead of raising an error",
            "Don't skip broken files instead of raising an error",
        ),
    }

    def setup(self):
        # prepare output path:
        if not self.output_path:
            self.log.critical("No output path was specified")
            sys.exit(1)

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
        if self.allowed_tels:
            self.first_subarray = self.first_subarray.select_subarray(
                tel_ids=self.allowed_tels
            )
            self.allowed_tel_names = {"tel_%03d" % i for i in self.allowed_tels}

        self.first_subarray.to_hdf(self.output_path)
        self.output_file = tables.open_file(self.output_path, mode="a")

        # setup required nodes
        self.usable_nodes = all_nodes

        if self.skip_simu_images is True:
            self.usable_nodes = self.usable_nodes - simu_images

        if self.skip_images is True:
            self.usable_nodes = self.usable_nodes - image_nodes

        if self.skip_parameters is True:
            self.usable_nodes = self.usable_nodes - parameter_nodes

    def check_file_broken(self, file):
        # Check that the file is not broken or any node is missing
        file_path = file.root._v_file.filename

        data_model_version = file.root._v_attrs[VERSION_KEY]
        if data_model_version != self.data_model_version:
            self.log.critical(
                f"File has data model version {data_model_version}"
                f", expected {self.data_model_version}"
            )
            return True

        current_subarray = SubarrayDescription.from_hdf(file_path)
        if self.allowed_tels:
            current_subarray = current_subarray.select_subarray(
                tel_ids=self.allowed_tels
            )
        broken = False

        # Check subarray
        if self.first_subarray != current_subarray:
            self.log.critical(f"Subarray does not match for {file_path}")
            broken = True

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
                broken = True

        return broken

    def add_image_statistics(self, file):
        # Creates table for image statistics and adds the entries together.
        # This does not append rows to the existing table

        if IMAGE_STATISTICS_PATH in self.output_file:
            table_out = self.output_file.root[IMAGE_STATISTICS_PATH]
            table_in = file.root[IMAGE_STATISTICS_PATH]

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

            for node in service_nodes:
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

                    if self.allowed_tels:
                        for tel_name in self.allowed_tel_names:
                            if tel_name in file.root[node]:
                                self._copy_or_append_tel_table(file, node, tel_name)

                        continue

                    for tel in file.root[node]:
                        self._copy_or_append_tel_table(file, node, tel.name)

                    continue

                if node in self.output_file:
                    data = file.root[node][:]
                    output_node = self.output_file.get_node(node)
                    output_node.append(data)
                else:
                    group_path, _ = os.path.split(node)
                    if group_path not in self.output_file:
                        self._create_group(group_path)

                    target_group = self.output_file.root[group_path]
                    file.copy_node(node, newparent=target_group)

    def _copy_or_append_tel_table(self, file, node, tel_name):
        tel_node_path = node + "/" + tel_name
        if tel_node_path in self.output_file:
            output_node = self.output_file.get_node(tel_node_path)
            input_node = file.root[tel_node_path]

            # cast needed for some image parameters that are sometimes
            # float32 and sometimes float64
            output_node.append(input_node[:].astype(output_node.dtype))
        else:
            target_group = self.output_file.root[node]
            file.copy_node(tel_node_path, newparent=target_group)

    def _create_group(self, node):
        head, tail = os.path.split(node)
        self.output_file.create_group(head, tail, createparents=True)

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

            if not DL1EventSource.is_compatible(current_file):
                self.log.critical(
                    f"input file {current_file} is not a supported DL1 file"
                )
                if self.skip_broken_files:
                    continue
                else:
                    sys.exit(1)

            with tables.open_file(current_file, mode="r") as file:
                if i == 0:
                    self.data_model_version = file.root._v_attrs[VERSION_KEY]

                    # Check if first file is simulation

                    if "/simulation" not in file.root:
                        self.usable_nodes = self.usable_nodes - simu_nodes
                        self.log.info("Merging real data")
                        self.is_simulation = False
                    else:
                        self.log.info("Merging simulation-files")
                        self.is_simulation = True

                if self.check_file_broken(file) is True:
                    if self.skip_broken_files is True:
                        continue
                    else:
                        self.log.critical("Broken file detected.")
                        sys.exit(1)

                self.merge_tables(file)
                if IMAGE_STATISTICS_PATH in file:
                    self.add_image_statistics(file)

            PROV.add_input_file(str(current_file))
            merged_files_counter += 1

        self.log.info(
            f"{merged_files_counter} out of {len(self.input_files)} files "
            "has been merged!"
        )

    def finish(self):
        self.output_file.close()
        activity = PROV.current_activity.provenance
        process_type_ = "Observation"
        if self.is_simulation is True:
            process_type_ = "Simulation"

        reference = meta.Reference(
            contact=meta.Contact(name="", email="", organization="CTA Consortium"),
            product=meta.Product(
                description="Merged DL1 Data Product",
                data_category="Sim",  # TODO: copy this from the inputs
                data_level=["DL1"],  # TODO: copy this from inputs
                data_association="Subarray",
                data_model_name="ASWG",  # TODO: copy this from inputs
                data_model_version=self.data_model_version,
                data_model_url="",
                format="hdf5",
            ),
            process=meta.Process(type_=process_type_, subtype="", id_="merge"),
            activity=meta.Activity.from_provenance(activity),
            instrument=meta.Instrument(
                site="Other",
                class_="Subarray",
                type_="unknown",
                version="unknown",
                id_=self.first_subarray.name,
            ),
        )

        headers = reference.to_dict()

        with HDF5TableWriter(
            self.output_path, parent=self, mode="a", add_prefix=True
        ) as writer:
            meta.write_to_hdf5(headers, writer.h5file)


def main():
    tool = MergeTool()
    tool.run()


if __name__ == "main":
    main()
