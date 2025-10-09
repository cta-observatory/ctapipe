import enum
import uuid
import warnings
from contextlib import ExitStack
from pathlib import Path

import tables
from astropy.time import Time

from ..containers import EventType
from ..core import Component, Provenance, traits
from ..instrument.optics import FocalLengthKind
from ..instrument.subarray import SubarrayDescription
from ..utils.arrays import recarray_drop_columns
from . import metadata
from .hdf5dataformat import (
    DL0_TEL_POINTING_GROUP,
    DL1_CAMERA_COEFFICIENTS_GROUP,
    DL1_COLUMN_NAMES,
    DL1_IMAGE_STATISTICS_TABLE,
    DL1_PIXEL_STATISTICS_GROUP,
    DL1_SUBARRAY_POINTING_GROUP,
    DL1_SUBARRAY_TRIGGER_TABLE,
    DL1_TEL_IMAGES_GROUP,
    DL1_TEL_MUON_GROUP,
    DL1_TEL_PARAMETERS_GROUP,
    DL1_TEL_POINTING_GROUP,
    DL1_TEL_TRIGGER_TABLE,
    DL2_EVENT_STATISTICS_GROUP,
    DL2_SUBARRAY_GROUP,
    DL2_TEL_GROUP,
    FIXED_POINTING_GROUP,
    OBSERVATION_BLOCK_TABLE,
    R0_TEL_GROUP,
    R1_TEL_GROUP,
    SCHEDULING_BLOCK_TABLE,
    SHOWER_DISTRIBUTION_TABLE,
    SIMULATION_GROUP,
    SIMULATION_IMAGES_GROUP,
    SIMULATION_IMPACT_GROUP,
    SIMULATION_PARAMETERS_GROUP,
    SIMULATION_RUN_TABLE,
    SIMULATION_SHOWER_TABLE,
)
from .hdf5tableio import DEFAULT_FILTERS, get_column_attrs, get_node_meta, split_h5path


class NodeType(enum.Enum):
    # a single table
    TABLE = enum.auto()
    # a group comprising tel_XXX tables
    TEL_GROUP = enum.auto()
    # a group with children of the form /<property group>/<iter table>
    ITER_GROUP = enum.auto()
    # a group with children of the form /<property group>/<iter group>/<tel_XXX table>
    ITER_TEL_GROUP = enum.auto()


#: nodes to check for merge-ability
_NODES_TO_CHECK = {
    SCHEDULING_BLOCK_TABLE: NodeType.TABLE,
    OBSERVATION_BLOCK_TABLE: NodeType.TABLE,
    SIMULATION_RUN_TABLE: NodeType.TABLE,
    FIXED_POINTING_GROUP: NodeType.TEL_GROUP,
    SIMULATION_GROUP: NodeType.TABLE,
    SHOWER_DISTRIBUTION_TABLE: NodeType.TABLE,
    SIMULATION_SHOWER_TABLE: NodeType.TABLE,
    SIMULATION_IMPACT_GROUP: NodeType.TEL_GROUP,
    SIMULATION_IMAGES_GROUP: NodeType.TEL_GROUP,
    SIMULATION_PARAMETERS_GROUP: NodeType.TEL_GROUP,
    R0_TEL_GROUP: NodeType.TEL_GROUP,
    R1_TEL_GROUP: NodeType.TEL_GROUP,
    DL1_SUBARRAY_TRIGGER_TABLE: NodeType.TABLE,
    DL1_TEL_TRIGGER_TABLE: NodeType.TABLE,
    DL1_TEL_IMAGES_GROUP: NodeType.TEL_GROUP,
    DL1_TEL_PARAMETERS_GROUP: NodeType.TEL_GROUP,
    DL1_TEL_MUON_GROUP: NodeType.TEL_GROUP,
    DL2_TEL_GROUP: NodeType.ITER_TEL_GROUP,
    DL2_SUBARRAY_GROUP: NodeType.ITER_GROUP,
    DL1_SUBARRAY_POINTING_GROUP: NodeType.TABLE,
    DL1_TEL_POINTING_GROUP: NodeType.TEL_GROUP,
}


def _get_required_nodes(h5file):
    """Return nodes to be required in a new file for appending to ``h5file``"""
    required_nodes = set()
    for node, node_type in _NODES_TO_CHECK.items():
        if node not in h5file.root:
            continue

        if node_type in (NodeType.TABLE, NodeType.TEL_GROUP):
            required_nodes.add(node)

        elif node_type is NodeType.ITER_GROUP:
            for kind_group in h5file.root[node]._f_iter_nodes("Group"):
                for table in kind_group._f_iter_nodes("Table"):
                    required_nodes.add(table._v_pathname)

        elif node_type is NodeType.ITER_TEL_GROUP:
            for kind_group in h5file.root[node]._f_iter_nodes("Group"):
                for iter_group in kind_group._f_iter_nodes("Group"):
                    required_nodes.add(iter_group._v_pathname)
        else:
            raise ValueError(f"Unhandled node type: {node_type} of {node}")

    return required_nodes


class CannotMerge(OSError):
    """Raised when trying to merge incompatible files"""


class HDF5Merger(Component):
    """
    Class to copy / append / merge ctapipe hdf5 files
    """

    output_path = traits.Path(directory_ok=False).tag(config=True)

    overwrite = traits.Bool(
        False,
        help="If true, the ``output_path`` is overwritten in case it exists. See also ``append``",
    ).tag(config=True)

    append = traits.Bool(
        False,
        help="If true, the ``output_path`` is appended to. See also ``overwrite``",
    ).tag(config=True)

    telescope_events = traits.Bool(
        True,
        help="Whether to include telescope-wise data in merged output",
    ).tag(config=True)

    simulation = traits.Bool(
        True,
        help="Whether to include data only known for simulations in merged output",
    ).tag(config=True)

    true_images = traits.Bool(
        True,
        help="Whether to include true images in merged output",
    ).tag(config=True)

    true_parameters = traits.Bool(
        True,
        help="Whether to include parameters calculated on true images in merged output",
    ).tag(config=True)

    r0_waveforms = traits.Bool(
        True,
        help="Whether to include r0 waveforms in merged output",
    ).tag(config=True)

    r1_waveforms = traits.Bool(
        True,
        help="Whether to include r1 waveforms in merged output",
    ).tag(config=True)

    dl1_images = traits.Bool(
        True,
        help="Whether to include dl1 images in merged output",
    ).tag(config=True)

    dl1_parameters = traits.Bool(
        True,
        help="Whether to include dl1 image parameters in merged output",
    ).tag(config=True)

    dl1_muon = traits.Bool(
        True,
        help="Whether to include dl1 muon parameters in merged output",
    ).tag(config=True)

    dl2_subarray = traits.Bool(
        True, help="Whether to include dl2 subarray-event-wise data in merged output"
    ).tag(config=True)

    dl2_telescope = traits.Bool(
        True, help="Whether to include dl2 telescope-event-wise data in merged output"
    ).tag(config=True)

    monitoring = traits.Bool(
        True, help="Whether to include monitoring data in merged output"
    ).tag(config=True)

    processing_statistics = traits.Bool(
        True, help="Whether to include processing statistics in merged output"
    ).tag(config=True)

    single_ob = traits.Bool(
        False,
        help=(
            "If true, input files are assumed to be multiple chunks from the same"
            " observation block and the ob / sb blocks will only be copied from "
            " the first input file"
        ),
    ).tag(config=True)

    def __init__(self, output_path=None, **kwargs):
        # enable using output_path as posarg
        if output_path not in {None, traits.Undefined}:
            kwargs["output_path"] = output_path

        super().__init__(**kwargs)

        if self.overwrite and self.append:
            raise traits.TraitError("overwrite and append are mutually exclusive")

        output_exists = self.output_path.exists()
        appending = False
        if output_exists and not (self.append or self.overwrite):
            raise traits.TraitError(
                f"output_path '{self.output_path}' exists but neither append nor overwrite allowed"
            )

        if output_exists and self.append:
            appending = True

        self.h5file = tables.open_file(
            self.output_path,
            mode="a" if appending else "w",
            filters=DEFAULT_FILTERS,
        )

        self.required_nodes = None
        self.data_model_version = None
        self.subarray = None
        self.meta = None
        self._merged_obs_ids = set()
        self._n_merged = 0

        # output file existed, so read subarray and data model version to make sure
        # any file given matches what we already have
        if appending:
            self.meta = self._read_meta(self.h5file)
            self.data_model_version = self.meta.product.data_model_version

            # focal length choice doesn't matter here, set to equivalent so we don't get
            # an error if only the effective focal length is available in the file
            self.subarray = SubarrayDescription.from_hdf(
                self.h5file,
                focal_length_choice=FocalLengthKind.EQUIVALENT,
            )
            self.required_nodes = _get_required_nodes(self.h5file)

            # this will update _merged_obs_ids from existing input file
            self._check_obs_ids(self.h5file)
            self._n_merged += 1

    def __call__(self, other: str | Path | tables.File):
        """
        Append file ``other`` to the output file
        """
        exit_stack = ExitStack()
        if not isinstance(other, tables.File):
            other = exit_stack.enter_context(tables.open_file(other, mode="r"))

        with exit_stack:
            # first file to be merged
            if self._n_merged == 0:
                self.meta = self._read_meta(other)
                self.data_model_version = self.meta.product.data_model_version
                metadata.write_to_hdf5(self.meta.to_dict(), self.h5file)
            else:
                self._check_can_merge(other)

            Provenance().add_input_file(other.filename, "data product to merge")
            try:
                self._append(other)
                # if first file, update required nodes
                if self.required_nodes is None:
                    self.required_nodes = _get_required_nodes(self.h5file)
                    self.log.info(
                        "Updated required nodes to %s", sorted(self.required_nodes)
                    )
                self._n_merged += 1
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

    def _read_meta(self, h5file):
        try:
            return metadata._read_reference_metadata_hdf5(h5file)
        except Exception:
            raise CannotMerge(
                f"CTAO Reference meta not found in input file: {h5file.filename}"
            )

    def _check_can_merge(self, other):
        other_meta = self._read_meta(other)
        other_version = other_meta.product.data_model_version
        if self.data_model_version != other_version:
            raise CannotMerge(
                f"Input file {other.filename:!r} has different data model version:"
                f" {other_version}, expected {self.data_model_version}"
            )

        for node_path in self.required_nodes:
            if node_path not in other.root:
                raise CannotMerge(
                    f"Required node {node_path} not found in {other.filename}"
                )

    def _check_obs_ids(self, other):
        keys = [OBSERVATION_BLOCK_TABLE, DL1_SUBARRAY_TRIGGER_TABLE]
        for key in keys:
            if key in other.root:
                obs_ids = other.root[key].col("obs_id")
                break
        else:
            raise CannotMerge(
                f"Input file {other.filename} is missing keys required to"
                f" check for duplicated obs_ids. Tried: {keys}"
            )

        if self.single_ob and len(self._merged_obs_ids) > 0:
            different = self._merged_obs_ids.symmetric_difference(obs_ids)
            if len(different) > 0:
                msg = f"Input file {other.filename} contains different obs_ids than already merged ({self._merged_obs_ids}) for single_ob=True: {different}"
                raise CannotMerge(msg)
        else:
            duplicated = self._merged_obs_ids.intersection(obs_ids)
            if len(duplicated) > 0:
                msg = f"Input file {other.filename} contains obs_ids already included in output file: {duplicated}"
                raise CannotMerge(msg)

        self._merged_obs_ids.update(obs_ids)

    def _append(self, other):
        self._check_obs_ids(other)

        # Configuration
        self._append_subarray(other)

        # in case of "single_ob", we only copy sb/ob blocks for the first file
        if not self.single_ob or self._n_merged == 0:
            config_keys = [SCHEDULING_BLOCK_TABLE, OBSERVATION_BLOCK_TABLE]
            for key in config_keys:
                if key in other.root:
                    self._append_table(other, other.root[key])

        if FIXED_POINTING_GROUP in other.root:
            self._append_table_group(
                other, other.root[FIXED_POINTING_GROUP], once=self.single_ob
            )

        # Simulation
        simulation_table_keys = [
            SIMULATION_RUN_TABLE,
            SHOWER_DISTRIBUTION_TABLE,
            SIMULATION_SHOWER_TABLE,
        ]
        for key in simulation_table_keys:
            if self.simulation and key in other.root:
                self._append_table(other, other.root[key])

        if (
            self.telescope_events
            and self.simulation
            and SIMULATION_IMPACT_GROUP in other.root
        ):
            self._append_table_group(other, other.root[SIMULATION_IMPACT_GROUP])

        if (
            self.telescope_events
            and self.simulation
            and SIMULATION_IMAGES_GROUP in other.root
        ):
            filter_columns = None if self.true_images else ["true_image"]
            self._append_table_group(
                other, other.root[SIMULATION_IMAGES_GROUP], filter_columns
            )

        if (
            self.telescope_events
            and self.simulation
            and self.true_parameters
            and SIMULATION_PARAMETERS_GROUP in other.root
        ):
            self._append_table_group(other, other.root[SIMULATION_PARAMETERS_GROUP])

        # R0
        if self.telescope_events and self.r0_waveforms and R0_TEL_GROUP in other.root:
            self._append_table_group(other, other.root[R0_TEL_GROUP])

        # R1
        if self.telescope_events and self.r1_waveforms and R1_TEL_GROUP in other.root:
            self._append_table_group(other, other.root[R1_TEL_GROUP])

        # DL1
        if DL1_SUBARRAY_TRIGGER_TABLE in other.root:
            self._append_table(other, other.root[DL1_SUBARRAY_TRIGGER_TABLE])

        if self.telescope_events and DL1_TEL_TRIGGER_TABLE in other.root:
            self._append_table(other, other.root[DL1_TEL_TRIGGER_TABLE])

        if (
            self.telescope_events
            and self.dl1_images
            and DL1_TEL_IMAGES_GROUP in other.root
        ):
            self._append_table_group(other, other.root[DL1_TEL_IMAGES_GROUP])

        if (
            self.telescope_events
            and self.dl1_parameters
            and DL1_TEL_PARAMETERS_GROUP in other.root
        ):
            self._append_table_group(other, other.root[DL1_TEL_PARAMETERS_GROUP])

        if self.telescope_events and self.dl1_muon and DL1_TEL_MUON_GROUP in other.root:
            self._append_table_group(other, other.root[DL1_TEL_MUON_GROUP])

        # DL2
        if self.telescope_events and self.dl2_telescope and DL2_TEL_GROUP in other.root:
            for kind_group in other.root[DL2_TEL_GROUP]._f_iter_nodes("Group"):
                for iter_group in kind_group._f_iter_nodes("Group"):
                    self._append_table_group(other, iter_group)

        if self.dl2_subarray and DL2_SUBARRAY_GROUP in other.root:
            for kind_group in other.root[DL2_SUBARRAY_GROUP]._f_iter_nodes("Group"):
                for table in kind_group._f_iter_nodes("Table"):
                    self._append_table(other, table)

        # Pointing monitoring
        if self.monitoring and DL1_SUBARRAY_POINTING_GROUP in other.root:
            self._append_table(other, other.root[DL1_SUBARRAY_POINTING_GROUP])

        if (
            self.monitoring
            and self.telescope_events
            and DL0_TEL_POINTING_GROUP in other.root
        ):
            self._append_table_group(other, other.root[DL0_TEL_POINTING_GROUP])

        if (
            self.monitoring
            and self.telescope_events
            and DL1_TEL_POINTING_GROUP in other.root
        ):
            self._append_table_group(other, other.root[DL1_TEL_POINTING_GROUP])

        # Calibration coefficients monitoring
        if (
            self.monitoring
            and self.telescope_events
            and DL1_CAMERA_COEFFICIENTS_GROUP in other.root
        ):
            self._append_table_group(other, other.root[DL1_CAMERA_COEFFICIENTS_GROUP])

        # Pixel statistics monitoring
        for dl1_colname in DL1_COLUMN_NAMES:
            for event_type in EventType:
                key = f"{DL1_PIXEL_STATISTICS_GROUP}/{event_type.name.lower()}_{dl1_colname}"
                if self.monitoring and self.telescope_events and key in other.root:
                    self._append_table_group(other, other.root[key])

        # quality query statistics
        if DL1_IMAGE_STATISTICS_TABLE in other.root:
            self._add_statistics_table(other, other.root[DL1_IMAGE_STATISTICS_TABLE])

        if DL2_EVENT_STATISTICS_GROUP in other.root:
            for node in other.root[DL2_EVENT_STATISTICS_GROUP]._f_iter_nodes("Table"):
                self._add_statistics_table(other, node)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if hasattr(self, "h5file"):
            self.h5file.close()
        Provenance().add_output_file(str(self.output_path))

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

    def _append_table_group(self, file, input_group, filter_columns=None, once=False):
        """Add a group that has a number of child tables to outputfile"""

        if not isinstance(input_group, tables.Group):
            raise TypeError(f"node must be a `tables.Group`, got {input_group}")

        node_path = input_group._v_pathname
        self._get_or_create_group(node_path)

        for table in input_group._f_iter_nodes("Table"):
            self._append_table(file, table, filter_columns=filter_columns, once=once)

    def _append_table(self, file, table, filter_columns=None, once=False):
        """Append a single table to the output file"""
        if not isinstance(table, tables.Table):
            raise TypeError(f"node must be a `tables.Table`, got {table}")

        table_path = table._v_pathname
        group_path, _ = split_h5path(table_path)

        if table_path in self.h5file:
            if once:
                return

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
                self._copy_node_filter_columns(table, filter_columns)

    def _copy_node_filter_columns(self, table, filter_columns):
        group_path, table_name = split_h5path(table._v_pathname)
        input_table = recarray_drop_columns(table[:], filter_columns)

        out_table = self.h5file.create_table(
            group_path,
            table_name,
            filters=table.filters,
            createparents=True,
            obj=input_table,
        )

        # copy metadata
        meta = get_node_meta(table)
        for key, val in meta.items():
            out_table.attrs[key] = val

        # set column attrs
        column_attrs = get_column_attrs(table)
        for pos, colname in enumerate(out_table.colnames):
            for key, value in column_attrs[colname].items():
                # these are taken from the table object itself, not actually from the attrs
                if key in {"POS", "DTYPE"}:
                    continue
                out_table.attrs[f"CTAFIELD_{pos}_{key}"] = value

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
