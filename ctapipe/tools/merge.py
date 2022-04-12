"""
Merge DL1-files from ctapipe-process tool
"""
import sys
import os
from argparse import ArgumentParser
from pathlib import Path
from traitlets import List

import tables
import numpy as np
from tqdm.auto import tqdm

from ..io import metadata as meta, HDF5EventSource, get_hdf5_datalevels
from ..io import HDF5TableWriter
from ..core import Provenance, Tool, traits
from ..core.traits import Bool, Set, Unicode, flag, CInt
from ..instrument import SubarrayDescription


PROV = Provenance()

VERSION_KEY = "CTA PRODUCT DATA MODEL VERSION"
IMAGE_STATISTICS_PATH = "/dl1/service/image_statistics"

required_nodes = {
    "/dl1/event/subarray/trigger",
    "/dl1/event/telescope/trigger",
    "/dl1/monitoring/subarray/pointing",
}

optional_nodes = {
    "/simulation/service/shower_distribution",
    "/simulation/event/telescope/images",
    "/simulation/event/telescope/parameters",
    "/dl1/event/telescope/parameters",
    "/dl1/event/telescope/images",
    "/dl1/service/image_statistics",
    "/dl1/service/image_statistics.__table_column_meta__",
    "/dl2/event/subarray/geometry",
}

simulation_nodes = {
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

simulation_images = {"/simulation/event/telescope/images"}

dl2_subarray_nodes = {"/dl2/event/subarray/geometry"}


all_nodes = (
    required_nodes
    | optional_nodes
    | simulation_nodes
    | service_nodes
    | nodes_with_tels
    | image_nodes
    | parameter_nodes
    | simulation_images
    | dl2_subarray_nodes
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
    overwrite = Bool(
        help="Overwrite output file if it exists", default_value=False
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

        # setup required nodes
        self.usable_nodes = all_nodes.copy()

        if self.skip_simu_images is True:
            self.usable_nodes -= simulation_images

        if self.skip_images is True:
            self.usable_nodes -= image_nodes

        if self.skip_parameters is True:
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

            # do not try to merge optional nodes not present in first file
            for node in filter(lambda n: n not in h5file, optional_nodes):
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
                if node in file:
                    file.copy_node(node, newparent=target_group)

    def _merge_tel_group(self, file, input_node):
        """Add a group that has one child table per telescope (type) to outputfile"""
        if not isinstance(input_node, tables.Group):
            raise TypeError(f"node must be a `tables.Group`, got {input_node}")

        node_path = input_node._v_pathname

        if input_node not in self.output_file:
            self._create_group(node_path)

        for tel_name, table in input_node._v_children.items():
            if not self.allowed_tels or tel_name in self.allowed_tel_names:
                self._merge_table(file, table)

    def _merge_table(self, file, input_node):
        if not isinstance(input_node, tables.Table):
            raise TypeError(f"node must be a `tables.Table`, got {input_node}")

        node_path = input_node._v_pathname
        if node_path in self.output_file:
            output_table = self.output_file.get_node(node_path)
            output_table.append(input_node[:].astype(output_table.dtype))
        else:
            group_path, _ = os.path.split(node_path)
            if group_path not in self.output_file:
                self._create_group(group_path)

            target_group = self.output_file.root[group_path]
            file.copy_node(node_path, newparent=target_group)

    def merge_tables(self, file):
        """Go over all mergeable nodes and append to outputfile"""
        for node_path in self.usable_nodes:

            # skip service nodes that should only be included from the first file
            if node_path in service_nodes:
                continue

            if node_path in file:
                node = file.root[node_path]

                # nodes with child tables per telescope (type)
                if node_path in nodes_with_tels:
                    self._merge_tel_group(file, node)

                # groups of tables (e.g. dl2)
                elif isinstance(node, tables.Group):
                    for table in node._v_children.values():
                        self._merge_table(file, table)

                # single table
                else:
                    self._merge_table(file, node)

    def _create_group(self, node):
        head, tail = os.path.split(node)
        self.output_file.create_group(head, tail, createparents=True)

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
                if IMAGE_STATISTICS_PATH in h5file:
                    self.add_image_statistics(h5file)

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
                data_level=datalevels,
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
