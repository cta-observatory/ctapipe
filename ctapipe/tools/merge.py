"""
Merge DL1-files from ctapipe-process tool
"""
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import tables
from tqdm.auto import tqdm
from traitlets import List

from ctapipe.utils.arrays import recarray_drop_columns

from ..core import Provenance, Tool, traits
from ..core.traits import Bool, CInt, Set, Unicode, flag
from ..instrument import SubarrayDescription
from ..io import HDF5EventSource, HDF5TableWriter, get_hdf5_datalevels
from ..io import metadata as meta

PROV = Provenance()

VERSION_KEY = "CTA PRODUCT DATA MODEL VERSION"
IMAGE_STATISTICS_PATH = "/dl1/service/image_statistics"
DL2_STATISTICS_GROUP = "/dl2/service/tel_event_statistics"

required_nodes = {
    "/dl1/event/subarray/trigger",
    "/dl1/event/telescope/trigger",
    "/dl1/monitoring/subarray/pointing",
}

optional_nodes = {
    "/simulation/service/shower_distribution",
    "/simulation/event/telescope/images",
    "/simulation/event/telescope/parameters",
    "/simulation/event/telescope/impact",
    "/dl1/event/telescope/parameters",
    "/dl1/event/telescope/images",
    "/dl2/event/telescope/geometry",
    "/dl2/event/telescope/impact",
    "/dl2/event/telescope/energy",
    "/dl2/event/telescope/classification",
    "/dl2/event/subarray/geometry",
    "/dl2/event/subarray/energy",
    "/dl2/event/subarray/classification",
}

observation_configuration_nodes = {
    "/configuration/observation/observation_block",
    "/configuration/observation/scheduling_block",
}

simulation_nodes = {
    "/simulation/event/subarray/shower",
    "/simulation/event/telescope/parameters",
    "/simulation/event/telescope/images",
    "/simulation/service/shower_distribution",
    "/configuration/simulation/run",
}
nodes_with_tels = {
    "/dl1/monitoring/telescope/pointing",
    "/dl1/event/telescope/parameters",
    "/dl1/event/telescope/images",
    "/simulation/event/telescope/parameters",
    "/simulation/event/telescope/images",
    "/simulation/event/telescope/impact",
}
image_nodes = {
    "/dl1/event/telescope/images",
}
parameter_nodes = {
    "/simulation/event/telescope/parameters",
    "/dl1/event/telescope/parameters",
}

SIMULATED_IMAGE_GROUP = "/simulation/event/telescope/images"
simulation_images = {SIMULATED_IMAGE_GROUP}

dl2_subarray_nodes = {"/dl2/event/subarray/geometry"}

dl2_algorithm_tel_nodes = {
    "/dl2/event/telescope/geometry",
    "/dl2/event/telescope/impact",
    "/dl2/event/telescope/energy",
    "/dl2/event/telescope/classification",
}

all_nodes = (
    required_nodes
    | optional_nodes
    | simulation_nodes
    | nodes_with_tels
    | image_nodes
    | parameter_nodes
    | simulation_images
    | dl2_subarray_nodes
    | observation_configuration_nodes
    | dl2_algorithm_tel_nodes
)


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
        help="Input dl1-directory",
        default_value=None,
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=False,
    ).tag(config=True)
    input_files = List(
        traits.Path(exists=True, directory_ok=False),
        default_value=[],
        help="Input dl1-files",
    ).tag(config=True)
    output_path = traits.Path(
        help="Merged-DL1 output filename", directory_ok=False, allow_none=False
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
    progress_bar = Bool(
        help="Show progress bar during processing", default_value=False
    ).tag(config=True)
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
        self.check_output(
            [
                self.output_path,
            ]
        )

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

        # setup required nodes
        self.usable_nodes = all_nodes.copy()

        if self.skip_simu_images:
            self.usable_nodes -= simulation_images

        if self.skip_images:
            self.usable_nodes -= image_nodes

        if self.skip_parameters:
            self.usable_nodes -= parameter_nodes

        # Use first file as reference to setup nodes to merge
        with tables.open_file(self.input_files[0], "r") as h5file:
            self.data_model_version = h5file.root._v_attrs[VERSION_KEY]

            # Check if first file is simulation
            if "/simulation" not in h5file.root:
                self.log.info("Merging observed data")
                self.usable_nodes -= simulation_nodes
                self.is_simulation = False
            else:
                self.log.info("Merging simulated data")
                self.is_simulation = True

            for node in optional_nodes:
                if node in self.usable_nodes and node not in h5file:
                    self.log.info(f"First file does not contain {node}, ignoring")
                    self.usable_nodes.remove(node)

        # create output file with subarray from first file
        self.first_subarray = SubarrayDescription.from_hdf(self.input_files[0])
        if self.allowed_tels:
            self.first_subarray = self.first_subarray.select_subarray(
                tel_ids=self.allowed_tels
            )
            self.allowed_tel_names = {
                f"tel_{tel_id:03d}" for tel_id in self.allowed_tels
            }

        self.first_subarray.to_hdf(self.output_path)
        self.output_file = tables.open_file(self.output_path, mode="a")

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

    def add_statistics(self, file):
        if IMAGE_STATISTICS_PATH in file.root:
            self.add_statistics_node(file, IMAGE_STATISTICS_PATH)

        if DL2_STATISTICS_GROUP in file.root:
            for node in file.root[DL2_STATISTICS_GROUP]._v_children.values():
                self.add_statistics_node(file, node._v_pathname)

    def add_statistics_node(self, file, node_path):
        """
        Creates table for image statistics and adds the entries together.

        This does not append rows to the existing table
        """

        table_in = file.root[node_path]

        if node_path in self.output_file.root:
            table_out = self.output_file.root[node_path]

            for col in ["counts", "cumulative_counts"]:
                table_out.modify_column(
                    colname=col,
                    column=table_out.col(col) + table_in.col(col),
                )
        else:
            self._copy_node(file, table_in)

    def _merge_tel_group(self, file, input_node, filter_columns=None):
        """Add a group that has one child table per telescope (type) to outputfile"""
        if not isinstance(input_node, tables.Group):
            raise TypeError(f"node must be a `tables.Group`, got {input_node}")

        node_path = input_node._v_pathname

        if input_node not in self.output_file:
            self._create_group(node_path)

        for tel_name, table in input_node._v_children.items():
            if not self.allowed_tels or tel_name in self.allowed_tel_names:
                self._merge_table(file, table, filter_columns=filter_columns)

    def _merge_table(self, file, input_node, filter_columns=None):
        if not isinstance(input_node, tables.Table):
            raise TypeError(f"node must be a `tables.Table`, got {input_node}")

        node_path = input_node._v_pathname
        if node_path in self.output_file:
            output_table = self.output_file.get_node(node_path)
            input_table = input_node[:]
            if filter_columns is not None:
                input_table = recarray_drop_columns(input_table, filter_columns)
            output_table.append(input_table.astype(output_table.dtype))
        else:
            group_path, _ = os.path.split(node_path)
            if group_path not in self.output_file:
                self._create_group(group_path)

            if filter_columns is None:
                self._copy_node(file, input_node)
            else:
                input_table = recarray_drop_columns(input_node[:], filter_columns)
                where, name = os.path.split(node_path)
                self.output_file.create_table(
                    where,
                    name,
                    filters=input_node.filters,
                    createparents=True,
                    obj=input_table,
                )

    def merge_tables(self, file):
        """Go over all mergeable nodes and append to outputfile"""
        for node_path in self.usable_nodes:
            if node_path in file:
                node = file.root[node_path]

                # nodes with child tables per telescope (type)
                if node_path in nodes_with_tels:
                    if node_path == SIMULATED_IMAGE_GROUP and self.skip_images:
                        filter_columns = ["true_image"]
                    else:
                        filter_columns = None
                    self._merge_tel_group(file, node, filter_columns=filter_columns)

                # nodes with subgroups which have telescope (type) children
                # here used for different DL2 algorithms
                elif node_path in dl2_algorithm_tel_nodes:
                    for _node in node._v_children.values():
                        self._merge_tel_group(file, _node)

                # groups of tables (e.g. dl2)
                elif isinstance(node, tables.Group):
                    for table in node._v_children.values():
                        self._merge_table(file, table)

                # single table
                else:
                    self._merge_table(file, node)

    def _create_group(self, node):
        head, tail = os.path.split(node)
        return self.output_file.create_group(head, tail, createparents=True)

    def _get_or_create_group(self, node):
        if node in self.output_file.root:
            return self.output_file.root[node]
        return self._create_group(node)

    def _copy_node(self, file, node):
        group_path, _ = os.path.split(node._v_pathname)
        target_group = self._get_or_create_group(group_path)
        file.copy_node(node, newparent=target_group)

    def start(self):
        merged_files_counter = 0

        for input_path in tqdm(
            self.input_files,
            desc="Merging",
            unit="Files",
            disable=not self.progress_bar,
        ):
            if not HDF5EventSource.is_compatible(input_path):
                self.log.critical(f"input file {input_path} is not a supported file")
                if self.skip_broken_files:
                    continue
                else:
                    sys.exit(1)

            with tables.open_file(input_path, mode="r") as h5file:

                if self.check_file_broken(h5file) is True:
                    if self.skip_broken_files is True:
                        continue
                    else:
                        self.log.critical("Broken file detected.")
                        sys.exit(1)

                self.merge_tables(h5file)
                self.add_statistics(h5file)

            PROV.add_input_file(str(input_path))
            merged_files_counter += 1

        self.log.info(
            f"{merged_files_counter} out of {len(self.input_files)} files "
            "has been merged!"
        )

    def finish(self):
        datalevels = [d.name for d in get_hdf5_datalevels(self.output_file)]
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
                data_levels=datalevels,
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
