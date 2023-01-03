from contextlib import ExitStack
from pathlib import Path
from typing import Union

import tables

from ctapipe.instrument.optics import FocalLengthKind
from ctapipe.instrument.subarray import SubarrayDescription

from ..core import Component, traits
from ..utils.arrays import recarray_drop_columns
from .hdf5tableio import DEFAULT_FILTERS


class CannotMerge(IOError):
    """Raised when trying to merge incompatible files"""


VERSION_KEY = "CTA PRODUCT DATA MODEL VERSION"


def split_h5path(path):
    """
    Split a path inside an hdf5 file into parent / child
    """
    head, _, tail = path.rstrip("/").rpartition("/")
    return head, tail


telescope_groups = {
    "/dl1/event/telescope/images",
    "/dl1/event/telescope/parameters",
    "/dl1/monitoring/telescope/pointing",
}

algorithm_groups = {
    "/dl2/event/subarray/geometry",
    "/dl2/event/subarray/energy",
    "/dl2/event/subarray/classification",
}

algorithm_tel_groups = {
    "/dl2/event/subarray/geometry",
    "/dl2/event/subarray/energy",
    "/dl2/event/subarray/classification",
}


class SelectMergeHDF5(Component):
    """
    Class to copy / append / merge ctapipe hdf5 files
    """

    output_path = traits.Path(directory_ok=False).tag(config=True)
    overwrite = traits.Bool(False).tag(config=True)

    telescope_events = traits.Bool(default_value=True).tag(config=True)

    dl1_images = traits.Bool(default_value=True).tag(config=True)
    dl1_parameters = traits.Bool(default_value=True).tag(config=True)

    true_images = traits.Bool(default_value=True).tag(config=True)
    true_parameters = traits.Bool(default_value=True).tag(config=True)

    def __init__(self, output_path=None, **kwargs):
        # enable using output_path as posarg
        if output_path not in {None, traits.Undefined}:
            kwargs["output_path"] = output_path

        super().__init__(**kwargs)

        if self.overwrite or not self.output_path.exists():
            mode = "w"
            self.subarray = None
            self.datamodel_version = None
        else:
            mode = "a"

        self.h5file = tables.open_file(
            self.output_path, mode=mode, filters=DEFAULT_FILTERS
        )

        if mode == "a":
            self.datamodel_version = self.h5file.root._v_attrs[VERSION_KEY]
            # focal length choice doesn't matter here, set to equivalent so we don't get
            # an error if only the effective focal length is available in the file
            self.subarray = SubarrayDescription.from_hdf(
                self.h5file, focal_length_choice=FocalLengthKind.EQUIVALENT
            )

    def __call__(self, other: Union[str, Path, tables.File]):
        """
        Append file ``other`` to the ouput file
        """
        stack = ExitStack()
        if not isinstance(other, tables.File):
            other = stack.enter_context(tables.open_file(other, mode="r"))

        with stack:
            self._append_subarray(other)

            key = "/dl1/event/subarray/trigger"
            self._append_table(other, other.root[key])

            key = "/dl1/event/telescope/trigger"
            if self.telescope_events and key in other.root:
                self.log.info("Appending %s", key)
                self._append_table(other, other.root[key])

            key = "/dl1/event/telescope/images"
            if self.telescope_events and self.dl1_images and key in other.root:
                self.log.info("Appending %s", key)
                self._append_table_group(other, other.root[key])

            key = "/dl1/event/telescope/parameters"
            if self.telescope_events and self.dl1_parameters and key in other.root:
                self.log.info("Appending %s", key)
                self._append_table_group(other, other.root[key])

            key = "/simulation/event/telescope/images"
            if self.telescope_events and key in other.root:
                self.log.info("Appending %s", key)
                filter_columns = None if self.true_images else ["true_image"]
                self._append_table_group(other, other.root[key], filter_columns)

            key = "/simulation/event/telescope/parameters"
            if self.telescope_events and self.true_parameters and key in other.root:
                self.log.info("Appending %s", key)
                self._append_table_group(other, other.root[key])

        self.h5file.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self, "h5file"):
            self.h5file.close()

    def _append_subarray(self, other):
        # focal length choice doesn't matter here, set to equivalent so we don't get
        # an error if only the effective focal length is available in the file
        subarray = SubarrayDescription.from_hdf(
            other, focal_length_choice=FocalLengthKind.EQUIVALENT
        )

        if self.subarray is None:
            self.subarray = subarray
            self.subarray.to_hdf(self.h5file)

        elif self.subarray != subarray:
            raise CannotMerge(f"Subarrays do not match for file: {other.filename}")

    def _append_table_group(self, file, input_node, filter_columns=None):
        """Add a group that has a number of child tables to outputfile"""

        if not isinstance(input_node, tables.Group):
            raise TypeError(f"node must be a `tables.Group`, got {input_node}")

        node_path = input_node._v_pathname

        if input_node not in self.h5file:
            self._create_group(node_path)

        for table in input_node._f_iter_nodes("Table"):
            self._append_table(file, table, filter_columns=filter_columns)

    def _append_table(self, file, input_node, filter_columns=None):
        """Append a single table to the output file"""
        if not isinstance(input_node, tables.Table):
            raise TypeError(f"node must be a `tables.Table`, got {input_node}")

        node_path = input_node._v_pathname
        group_path, node_name = split_h5path(node_path)

        if node_path in self.h5file:
            output_table = self.h5file.get_node(node_path)
            input_table = input_node[:]
            if filter_columns is not None:
                input_table = recarray_drop_columns(input_table, filter_columns)

            output_table.append(input_table.astype(output_table.dtype))

        else:
            if group_path not in self.h5file:
                self._create_group(group_path)

            if filter_columns is None:
                self._copy_node(file, input_node)
            else:
                input_table = recarray_drop_columns(input_node[:], filter_columns)
                self.h5file.create_table(
                    group_path,
                    node_name,
                    filters=input_node.filters,
                    createparents=True,
                    obj=input_table,
                )

    def _create_group(self, node):
        parent, name = split_h5path(node)
        return self.h5file.create_group(parent, name, createparents=True)

    def _get_or_create_group(self, node):
        if node in self.h5file.root:
            return self.h5file.root[node]
        return self._create_group(node)

    def _copy_node(self, file, node):
        group_path, _ = split_h5path(node._v_pathname)
        target_group = self._get_or_create_group(group_path)
        file.copy_node(node, newparent=target_group)
