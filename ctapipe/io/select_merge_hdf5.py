import warnings
from contextlib import ExitStack
from pathlib import Path
from typing import Union

import tables
from astropy.time import Time
from numba.core.extending import uuid

from ..core import Component, Provenance, traits
from ..instrument.optics import FocalLengthKind
from ..instrument.subarray import SubarrayDescription
from ..utils.arrays import recarray_drop_columns
from . import metadata
from .hdf5tableio import DEFAULT_FILTERS


class CannotMerge(IOError):
    """Raised when trying to merge incompatible files"""


def split_h5path(path):
    """
    Split a path inside an hdf5 file into parent / child
    """
    head, _, tail = path.rstrip("/").rpartition("/")
    return head, tail


class HDF5Merger(Component):
    """
    Class to copy / append / merge ctapipe hdf5 files
    """

    output_path = traits.Path(directory_ok=False).tag(config=True)

    overwrite = traits.Bool(False).tag(config=True)
    append = traits.Bool(False).tag(config=True)

    telescope_events = traits.Bool(default_value=True).tag(config=True)

    simulation = traits.Bool(True).tag(config=True)
    true_images = traits.Bool(True).tag(config=True)
    true_parameters = traits.Bool(True).tag(config=True)

    dl1_images = traits.Bool(True).tag(config=True)
    dl1_parameters = traits.Bool(True).tag(config=True)

    dl2_subarray = traits.Bool(True).tag(config=True)
    dl2_telescope = traits.Bool(True).tag(config=True)

    monitoring = traits.Bool(True).tag(config=True)

    statistics = traits.Bool(True).tag(config=True)

    def __init__(self, output_path=None, **kwargs):
        # enable using output_path as posarg
        if output_path not in {None, traits.Undefined}:
            kwargs["output_path"] = output_path
        super().__init__(**kwargs)

        output_exists = self.output_path.exists()

        if self.overwrite and self.append:
            raise traits.TraitError("overwrite and append are mutually exclusive")

        if not self.append and not self.overwrite and output_exists:
            raise traits.TraitError(
                f"output_path '{self.output_path}' exists but neither append nor overwrite allowed"
            )

        if self.overwrite:
            output_exists = False

        self.h5file = tables.open_file(
            self.output_path,
            mode="w" if self.overwrite else "a",
            filters=DEFAULT_FILTERS,
        )
        Provenance().add_output_file(str(self.output_path))

        self.data_model_version = None
        self.subarray = None
        self.meta = None
        # output file existed, so read subarray and data model version to make sure
        # any file given matches what we already have
        if output_exists:
            try:
                self.meta = metadata.Reference.from_dict(
                    metadata.read_metadata(self.h5file)
                )
            except Exception:
                raise CannotMerge(
                    f"CTA Rerence meta not found in existing output file: {self.output_path}"
                )

            self.data_model_version = self.meta.product.data_model_version

            # focal length choice doesn't matter here, set to equivalent so we don't get
            # an error if only the effective focal length is available in the file
            self.subarray = SubarrayDescription.from_hdf(
                self.h5file, focal_length_choice=FocalLengthKind.EQUIVALENT
            )

    def __call__(self, other: Union[str, Path, tables.File]):
        """
        Append file ``other`` to the ouput file
        """
        exit_stack = ExitStack()
        if not isinstance(other, tables.File):
            other = exit_stack.enter_context(tables.open_file(other, mode="r"))

        with exit_stack:
            self._check_can_merge(other)

            Provenance().add_input_file(other.filename)
            try:
                self._append(other)
            finally:
                self._update_meta()

    def _update_meta(self):
        # update creation date and id
        time = Time.now()
        id_ = str(uuid.uuid4())
        self.meta.product.id_ = id_
        self.meta.product.creation_time = time

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", tables.NaturalNameWarning)
            self.h5file.root._v_attrs["CTA PRODUCT CREATION TIME"] = time.iso
            self.h5file.root._v_attrs["CTA PRODUCT ID"] = id_
        self.h5file.flush()

    def _check_can_merge(self, other):
        try:
            other_meta = metadata.Reference.from_dict(metadata.read_metadata(other))
        except Exception:
            raise CannotMerge(
                f"CTA Rerence meta not found in input file: {other.filename}"
            )

        if self.meta is None:
            self.meta = other_meta
            self.data_model_version = self.meta.product.data_model_version
            metadata.write_to_hdf5(self.meta.to_dict(), self.h5file)
        else:
            other_version = other_meta.product.data_model_version
            if self.data_model_version != other_version:
                raise CannotMerge(
                    f"Input file {other.filename:!r} has different data model version:"
                    f" {other_version}, expected {self.data_model_version}"
                )

    def _append(self, other):
        # Configuration
        self._append_subarray(other)

        key = "/configuration/observation/scheduling_block"
        if key in other.root:
            self._append_table(other, other.root[key])

        key = "/configuration/observation/observation_block"
        if key in other.root:
            self._append_table(other, other.root[key])

        key = "/configuration/simulation/run"
        if self.simulation and key in other.root:
            self._append_table(other, other.root[key])

        # Simulation
        key = "/simulation/service/shower_distribution"
        if self.simulation and key in other.root:
            self._append_table(other, other.root[key])

        key = "/simulation/event/subarray/shower"
        if self.simulation and key in other.root:
            self._append_table(other, other.root[key])

        key = "/simulation/event/telescope/impact"
        if self.telescope_events and self.simulation and key in other.root:
            self._append_table_group(other, other.root[key])

        key = "/simulation/event/telescope/images"
        if self.telescope_events and self.simulation and key in other.root:
            self.log.info("Appending %s", key)
            filter_columns = None if self.true_images else ["true_image"]
            self._append_table_group(other, other.root[key], filter_columns)

        key = "/simulation/event/telescope/parameters"
        if (
            self.telescope_events
            and self.simulation
            and self.true_parameters
            and key in other.root
        ):
            self._append_table_group(other, other.root[key])

        # DL1
        key = "/dl1/event/subarray/trigger"
        if key in other.root:
            self._append_table(other, other.root[key])

        key = "/dl1/event/telescope/trigger"
        if self.telescope_events and key in other.root:
            self._append_table(other, other.root[key])

        key = "/dl1/event/telescope/images"
        if self.telescope_events and self.dl1_images and key in other.root:
            self._append_table_group(other, other.root[key])

        key = "/dl1/event/telescope/parameters"
        if self.telescope_events and self.dl1_parameters and key in other.root:
            self._append_table_group(other, other.root[key])

        # DL2
        key = "/dl2/event/telescope"
        if self.telescope_events and self.dl2_telescope and key in other.root:
            for kind_group in other.root[key]._f_iter_nodes("Group"):
                for algorithm_group in kind_group._f_iter_nodes("Group"):
                    self._append_table_group(other, algorithm_group)

        key = "/dl2/event/subarray"
        if self.dl2_subarray and key in other.root:
            for kind_group in other.root[key]._f_iter_nodes("Group"):
                for table in kind_group._f_iter_nodes("Table"):
                    self._append_table(other, table)

        # monitoring
        key = "/dl1/monitoring/subarray/pointing"
        if self.monitoring and key in other.root:
            self._append_table(other, other.root[key])

        key = "/dl1/monitoring/telescope/pointing"
        if self.monitoring and self.telescope_events and key in other.root:
            self._append_table_group(other, other.root[key])

        # quality query statistics
        key = "/dl1/service/image_statistics"
        if key in other.root:
            self._add_statistics_table(other, other.root[key])

        key = "/dl2/service/tel_event_statistics"
        if key in other.root:
            for node in other.root[key]._f_iter_nodes("Table"):
                self._add_statistics_table(other, node)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
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

    def _append_table_group(self, file, input_group, filter_columns=None):
        """Add a group that has a number of child tables to outputfile"""

        if not isinstance(input_group, tables.Group):
            raise TypeError(f"node must be a `tables.Group`, got {input_group}")

        node_path = input_group._v_pathname
        self._get_or_create_group(node_path)

        for table in input_group._f_iter_nodes("Table"):
            self._append_table(file, table, filter_columns=filter_columns)

    def _append_table(self, file, table, filter_columns=None):
        """Append a single table to the output file"""
        if not isinstance(table, tables.Table):
            raise TypeError(f"node must be a `tables.Table`, got {table}")

        table_path = table._v_pathname
        group_path, table_name = split_h5path(table_path)

        if table_path in self.h5file:
            output_table = self.h5file.get_node(table_path)
            input_table = table[:]
            if filter_columns is not None:
                input_table = recarray_drop_columns(input_table, filter_columns)

            output_table.append(input_table.astype(output_table.dtype))

        else:
            self._get_or_create_group(group_path)

            if filter_columns is None:
                self._copy_node(file, table)
            else:
                input_table = recarray_drop_columns(table[:], filter_columns)
                self.h5file.create_table(
                    group_path,
                    table_name,
                    filters=table.filters,
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

    def _add_statistics_table(self, file: tables.File, input_table: tables.Table):
        """
        Creates table for image statistics and adds the entries together.

        This does not append rows to the existing table
        """
        if not isinstance(input_table, tables.Table):
            raise TypeError(f"node must be a `tables.Table`, got {input_table}")

        table_path = input_table._v_pathname
        if table_path in self.h5file.root:
            table_out = self.h5file.root[table_path]

            for col in ["counts", "cumulative_counts"]:
                table_out.modify_column(
                    colname=col,
                    column=table_out.col(col) + input_table.col(col),
                )
        else:
            self._copy_node(file, input_table)
