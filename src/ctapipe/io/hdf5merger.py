import enum
import uuid
import warnings
from contextlib import ExitStack
from pathlib import Path

import tables
from astropy.table import join, unique, vstack
from astropy.time import Time

from ..containers import EventType
from ..core import Component, Provenance, traits
from ..instrument.optics import FocalLengthKind
from ..instrument.subarray import SubarrayDescription
from ..utils.arrays import recarray_drop_columns
from . import metadata, read_table, write_table
from .hdf5dataformat import (
    DL0_TEL_POINTING_GROUP,
    DL1_CAMERA_COEFFICIENTS_GROUP,
    DL1_COLUMN_NAMES,
    DL1_IMAGE_STATISTICS_TABLE,
    DL1_PIXEL_STATISTICS_GROUP,
    DL1_SUBARRAY_POINTING_GROUP,
    DL1_SUBARRAY_TRIGGER_TABLE,
    DL1_TEL_CALIBRATION_GROUP,
    DL1_TEL_ILLUMINATOR_THROUGHPUT_GROUP,
    DL1_TEL_IMAGES_GROUP,
    DL1_TEL_MUON_GROUP,
    DL1_TEL_MUON_THROUGHPUT_GROUP,
    DL1_TEL_OPTICAL_PSF_GROUP,
    DL1_TEL_PARAMETERS_GROUP,
    DL1_TEL_POINTING_GROUP,
    DL1_TEL_TRIGGER_TABLE,
    DL2_EVENT_STATISTICS_GROUP,
    DL2_SUBARRAY_CROSS_CALIBRATION_GROUP,
    DL2_SUBARRAY_GROUP,
    DL2_SUBARRAY_INTER_CALIBRATION_GROUP,
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

COMPATIBLE_DATA_MODEL_VERSIONS = [
    "v7.2.0",
    "v7.3.0",
]
SUBARRAY_EVENT_KEYS = ["obs_id", "event_id"]
TEL_EVENT_KEYS = ["obs_id", "event_id", "tel_id"]


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
    DL1_SUBARRAY_POINTING_GROUP: NodeType.TABLE,
    DL1_TEL_POINTING_GROUP: NodeType.TEL_GROUP,
    DL1_PIXEL_STATISTICS_GROUP: NodeType.ITER_TEL_GROUP,
    DL1_CAMERA_COEFFICIENTS_GROUP: NodeType.TEL_GROUP,
    DL1_TEL_MUON_THROUGHPUT_GROUP: NodeType.TEL_GROUP,
    DL2_TEL_GROUP: NodeType.ITER_TEL_GROUP,
    DL2_SUBARRAY_GROUP: NodeType.ITER_GROUP,
    DL2_SUBARRAY_CROSS_CALIBRATION_GROUP: NodeType.TABLE,
    DL2_SUBARRAY_INTER_CALIBRATION_GROUP: NodeType.TABLE,
}


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

    merge_strategy = traits.CaselessStrEnum(
        [
            "events-multiple-obs",
            "events-single-ob",
            "combine-telescope-data",
            "monitoring-only",
        ],
        default_value="events-multiple-obs",
        help=(
            "Strategy to handle different use cases when merging HDF5 files. "
            "'events-multiple-obs': allows merging event files (w and w/o monitoring data) from different observation blocks; "
            "'events-single-ob': for merging events in consecutive chunks of the same OB; "
            "'combine-telescope-data': merges telescope-wise data from different files for the same OB (requires telescope_events=True); "
            "'monitoring-only': attaches horizontally monitoring data from the same OB (requires monitoring=True)."
        ),
    ).tag(config=True)

    def __init__(self, output_path=None, **kwargs):
        # enable using output_path as posarg
        if output_path not in {None, traits.Undefined}:
            kwargs["output_path"] = output_path

        super().__init__(**kwargs)

        if self.overwrite and self.append:
            raise traits.TraitError("overwrite and append are mutually exclusive")

        # set convenient flags based on merge strategy
        self.single_ob = (
            self.merge_strategy == "events-single-ob"
            or self.merge_strategy == "monitoring-only"
            or self.merge_strategy == "combine-telescope-data"
        )
        self.attach_monitoring = self.merge_strategy == "monitoring-only"
        if self.attach_monitoring and not self.monitoring:
            raise traits.TraitError(
                "Merge strategy 'monitoring-only' requires monitoring=True"
            )
        self.combine_telescope_data = self.merge_strategy == "combine-telescope-data"
        if self.combine_telescope_data and (
            not self.telescope_events or self.dl2_telescope or self.dl2_subarray
        ):
            raise traits.TraitError(
                "Merge strategy 'combine-telescope-data' requires telescope_events=True "
                "and dl2_telescope=False and dl2_subarray=False"
            )

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
        self.data_category = None
        self.subarray = None
        self.subarray_list = []
        self.tel_trigger_tables, self.shower_tables = [], []
        self.tel_ids_set = set()
        self.meta = None
        self._merged_obs_ids = set()
        self._n_merged = 0

        # output file existed, so read subarray and data model version to make sure
        # any file given matches what we already have
        if appending:
            self.meta = self._read_meta(self.h5file)
            self.data_model_version = self.meta.product.data_model_version
            self.data_category = self.meta.product.data_category

            # focal length choice doesn't matter here, set to equivalent so we don't get
            # an error if only the effective focal length is available in the file
            self.subarray = SubarrayDescription.from_hdf(
                self.h5file,
                focal_length_choice=FocalLengthKind.EQUIVALENT,
            )
            self.subarray_list.append(self.subarray)

            # Update tel_ids_set with existing telescope IDs when appending
            if self.combine_telescope_data:
                self.tel_ids_set.update(self.subarray.tel_ids)

            # Get required nodes from existing output file
            self.required_nodes = self._get_required_nodes(self.h5file)

            # this will update _merged_obs_ids from existing input file
            self._check_obs_ids(self.h5file)

            # Append existing tel trigger tables and shower tables
            # if the merge strategy is 'combine-telescope-data'
            if self.combine_telescope_data:
                self.tel_trigger_tables.append(
                    read_table(self.h5file, DL1_TEL_TRIGGER_TABLE)
                )
                self.shower_tables.append(
                    read_table(self.h5file, SIMULATION_SHOWER_TABLE)
                )
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
                self.data_category = self.meta.product.data_category
                metadata.write_to_hdf5(self.meta.to_dict(), self.h5file)
            else:
                self._check_can_merge(other)

            Provenance().add_input_file(other.filename, "data product to merge")
            try:
                self._append(other)
                # if first file, update required nodes
                if self.required_nodes is None:
                    self.required_nodes = self._get_required_nodes(self.h5file)
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
        if self.attach_monitoring:
            if other_version not in COMPATIBLE_DATA_MODEL_VERSIONS:
                raise CannotMerge(
                    f"Input file {other.filename!r} has incompatible data model version"
                    f" for attaching monitoring data: {other_version}, expected one of"
                    f" {COMPATIBLE_DATA_MODEL_VERSIONS}"
                )
        else:
            if self.data_model_version != other_version:
                raise CannotMerge(
                    f"Input file {other.filename!r} has different data model version:"
                    f" {other_version}, expected {self.data_model_version}"
                )
        other_category = other_meta.product.data_category
        if self.data_category != other_category:
            raise CannotMerge(
                f"Input file {other.filename!r} has different data category:"
                f" {other_category}, expected {self.data_category}"
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
            # If monitoring data from the same observation block is being attached,
            # obs_ids can be different in case of MC simulations.
            if len(different) > 0 and self.data_category != "Sim":
                msg = (
                    f"Merge strategy '{self.merge_strategy}' selected, but input file {other.filename} contains "
                    f"different obs_ids than already merged ({self._merged_obs_ids}): {different}"
                )
                raise CannotMerge(msg)
        else:
            duplicated = self._merged_obs_ids.intersection(obs_ids)
            if len(duplicated) > 0:
                msg = f"Input file {other.filename} contains obs_ids already included in output file: {duplicated}"
                raise CannotMerge(msg)

        self._merged_obs_ids.update(obs_ids)

    def _get_required_nodes(self, h5file):
        """Return nodes to be required in a new file for appending to ``h5file``"""
        required_nodes = set()
        # Required nodes are not relevant for attaching monitoring data.
        if self.attach_monitoring:
            self.log.info("No required nodes to check for attaching monitoring data.")
            return required_nodes
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
        self.log.info("Updated required nodes to %s", sorted(required_nodes))
        return required_nodes

    def _append_simulation_data(self, other):
        """Append simulation-related data (run, shower, impact, images, parameters)."""

        if SIMULATION_SHOWER_TABLE in other.root:
            if self.combine_telescope_data:
                self.shower_tables.append(read_table(other, SIMULATION_SHOWER_TABLE))
            else:
                self._append_table(other, other.root[SIMULATION_SHOWER_TABLE])
        simulation_table_keys = [
            SIMULATION_RUN_TABLE,
            SHOWER_DISTRIBUTION_TABLE,
        ]
        for key in simulation_table_keys:
            if key in other.root:
                self._append_table(
                    other, other.root[key], once=self.combine_telescope_data
                )

        if FIXED_POINTING_GROUP in other.root:
            self._append_table_group(
                other,
                other.root[FIXED_POINTING_GROUP],
                once=self.single_ob,
            )

        if not self.telescope_events:
            return
        if SIMULATION_IMPACT_GROUP in other.root:
            self._append_table_group(other, other.root[SIMULATION_IMPACT_GROUP])

        if SIMULATION_IMAGES_GROUP in other.root:
            filter_columns = None if self.true_images else ["true_image"]
            self._append_table_group(
                other, other.root[SIMULATION_IMAGES_GROUP], filter_columns
            )

        if self.true_parameters and SIMULATION_PARAMETERS_GROUP in other.root:
            self._append_table_group(other, other.root[SIMULATION_PARAMETERS_GROUP])

    def _append_waveform_data(self, other):
        """Append R0 and R1 waveform data."""
        # R0
        if self.r0_waveforms and R0_TEL_GROUP in other.root:
            self._append_table_group(other, other.root[R0_TEL_GROUP])

        # R1
        if self.r1_waveforms and R1_TEL_GROUP in other.root:
            self._append_table_group(other, other.root[R1_TEL_GROUP])

    def _append_dl1_data(self, other):
        """Append DL1 data (triggers, images, parameters, muon)."""
        if DL1_SUBARRAY_TRIGGER_TABLE in other.root and not self.combine_telescope_data:
            self._append_table(other, other.root[DL1_SUBARRAY_TRIGGER_TABLE])

        if not self.telescope_events:
            return

        if DL1_TEL_TRIGGER_TABLE in other.root:
            if self.combine_telescope_data:
                self.tel_trigger_tables.append(read_table(other, DL1_TEL_TRIGGER_TABLE))
            else:
                self._append_table(other, other.root[DL1_TEL_TRIGGER_TABLE])

        if self.dl1_images and DL1_TEL_IMAGES_GROUP in other.root:
            self._append_table_group(other, other.root[DL1_TEL_IMAGES_GROUP])

        if self.dl1_parameters and DL1_TEL_PARAMETERS_GROUP in other.root:
            self._append_table_group(other, other.root[DL1_TEL_PARAMETERS_GROUP])

        if self.dl1_muon and DL1_TEL_MUON_GROUP in other.root:
            self._append_table_group(other, other.root[DL1_TEL_MUON_GROUP])

    def _append_dl2_data(self, other):
        """Append DL2 data (telescope and subarray events)."""
        # DL2 telescope data
        if self.telescope_events and self.dl2_telescope and DL2_TEL_GROUP in other.root:
            for kind_group in other.root[DL2_TEL_GROUP]._f_iter_nodes("Group"):
                for iter_group in kind_group._f_iter_nodes("Group"):
                    self._append_table_group(other, iter_group)

        # DL2 subarray data
        if self.dl2_subarray and DL2_SUBARRAY_GROUP in other.root:
            for kind_group in other.root[DL2_SUBARRAY_GROUP]._f_iter_nodes("Group"):
                for table in kind_group._f_iter_nodes("Table"):
                    self._append_table(other, table)

    def _append_monitoring_data(self, other):
        """Append monitoring data (pointing, calibration, throughput, pixel statistics)."""
        self._append_monitoring_subarray_groups(other)
        self._append_monitoring_dl2_groups(other)
        if self.telescope_events:
            self._append_monitoring_telescope_groups(other)
            self._append_pixel_statistics(other)

    def _append_monitoring_subarray_groups(self, other):
        """Append monitoring subarray groups."""
        monitoring_dl1_subarray_groups = [
            DL1_SUBARRAY_POINTING_GROUP,
        ]
        for key in monitoring_dl1_subarray_groups:
            if key in other.root:
                self._append_table(other, other.root[key], once=self.single_ob)

    def _append_monitoring_telescope_groups(self, other):
        """Append monitoring telescope groups."""
        monitoring_telescope_groups = [
            DL0_TEL_POINTING_GROUP,
            DL1_TEL_POINTING_GROUP,
            DL1_TEL_OPTICAL_PSF_GROUP,
            DL1_TEL_CALIBRATION_GROUP,
            DL1_CAMERA_COEFFICIENTS_GROUP,
            DL1_TEL_MUON_THROUGHPUT_GROUP,
            DL1_TEL_ILLUMINATOR_THROUGHPUT_GROUP,
        ]
        for key in monitoring_telescope_groups:
            if key in other.root:
                self._append_table_group(other, other.root[key], once=self.single_ob)

    def _append_monitoring_dl2_groups(self, other):
        """Append monitoring DL2 subarray groups."""
        monitoring_dl2_subarray_groups = [
            DL2_SUBARRAY_INTER_CALIBRATION_GROUP,
            DL2_SUBARRAY_CROSS_CALIBRATION_GROUP,
        ]
        for key in monitoring_dl2_subarray_groups:
            if self.dl2_subarray and key in other.root:
                self._append_table(other, other.root[key], once=self.single_ob)

    def _append_pixel_statistics(self, other):
        """Append pixel statistics monitoring data."""
        for dl1_colname in DL1_COLUMN_NAMES:
            for event_type in EventType:
                key = f"{DL1_PIXEL_STATISTICS_GROUP}/{event_type.name.lower()}_{dl1_colname}"
                if key in other.root:
                    self._append_table_group(
                        other, other.root[key], once=self.single_ob
                    )

    def _append_statistics_data(self, other):
        """Append processing statistics data."""
        # quality query statistics
        if DL1_IMAGE_STATISTICS_TABLE in other.root:
            self._add_statistics_table(other, other.root[DL1_IMAGE_STATISTICS_TABLE])

        if DL2_EVENT_STATISTICS_GROUP in other.root:
            for node in other.root[DL2_EVENT_STATISTICS_GROUP]._f_iter_nodes("Table"):
                self._add_statistics_table(other, node)

    def _append(self, other):
        """Append data to the output file."""
        self._check_obs_ids(other)
        self._append_subarray(other)
        self._append_configuration(other)
        if self.simulation and not self.attach_monitoring:
            self._append_simulation_data(other)
        if self.telescope_events and not self.attach_monitoring:
            self._append_waveform_data(other)
        if not self.attach_monitoring:
            self._append_dl1_data(other)
            self._append_dl2_data(other)
        if self.monitoring:
            self._append_monitoring_data(other)
        if self.processing_statistics and not self.attach_monitoring:
            self._append_statistics_data(other)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        self._flush()

    def _flush(self):
        """Flush any remaining data to the output file.

        This is relevant for the 'combine-telescope-data' merge strategy,
        where subarray and tel_trigger tables are only written at the end
        after all files have been processed.
        """

        if not self.combine_telescope_data or not self.tel_trigger_tables:
            return

        # Merge all subarrays into one
        merged_subarray = SubarrayDescription.merge_subarrays(self.subarray_list)
        # Write merged subarray to HDF5 (overwrite if existing)
        merged_subarray.to_hdf(self.output_path, overwrite=True)
        # Stack the tel_trigger tables vertically and sort by telescope event keys
        combined_tel_triggers = vstack(self.tel_trigger_tables)
        combined_tel_triggers.sort(TEL_EVENT_KEYS)
        # Write combined telescope trigger table to HDF5 (overwrite if existing)
        write_table(
            combined_tel_triggers,
            self.output_path,
            path=DL1_TEL_TRIGGER_TABLE,
            overwrite=True,
        )
        # Create the subarray trigger table from the combined telescope triggers
        subarray_trigger_table = combined_tel_triggers.copy()
        subarray_trigger_table.keep_columns(
            SUBARRAY_EVENT_KEYS + ["time", "event_type"]
        )
        subarray_trigger_table = unique(
            subarray_trigger_table, keys=SUBARRAY_EVENT_KEYS
        )
        # Add tels_with_trigger column indicating which telescopes had a trigger for each event
        tel_trigger_groups = combined_tel_triggers.group_by(SUBARRAY_EVENT_KEYS)
        tel_with_trigger = []
        for tel_trigger in tel_trigger_groups.groups:
            tel_with_trigger.append(
                merged_subarray.tel_ids_to_mask(tel_trigger["tel_id"])
            )
        # Add the new column to the table indicating which telescopes had a trigger for each event
        subarray_trigger_table.add_column(
            tel_with_trigger, index=-2, name="tels_with_trigger"
        )
        # Write subarray trigger table to HDF5 (overwrite if existing)
        write_table(
            subarray_trigger_table,
            self.output_path,
            DL1_SUBARRAY_TRIGGER_TABLE,
            overwrite=True,
        )
        # Create and write the merged shower table with only unique events
        # that are also in the subarray trigger table
        if self.shower_tables:
            # Stack the shower tables vertically and keep only unique events
            shower_table_stacked = vstack(self.shower_tables)
            shower_table_stacked = unique(
                shower_table_stacked, keys=SUBARRAY_EVENT_KEYS
            )
            # Join with subarray trigger table to keep only events that had a trigger
            shower_table = join(
                shower_table_stacked,
                subarray_trigger_table[SUBARRAY_EVENT_KEYS],
                join_type="inner",
            )
            shower_table.sort(SUBARRAY_EVENT_KEYS)
            # Write shower table to HDF5 (overwrite if existing)
            write_table(
                shower_table,
                self.output_path,
                SIMULATION_SHOWER_TABLE,
                overwrite=True,
            )

    def close(self):
        if hasattr(self, "h5file"):
            self.h5file.close()
        Provenance().add_output_file(str(self.output_path))

    def _append_configuration(self, other):
        """Append configuration-related data (scheduling blocks, observation blocks, pointing)."""
        # in case of "single_ob", we only copy sb/ob blocks for the first file
        if not self.single_ob or self._n_merged == 0:
            config_keys = [SCHEDULING_BLOCK_TABLE, OBSERVATION_BLOCK_TABLE]
            for key in config_keys:
                if key in other.root:
                    self._append_table(other, other.root[key])

    def _append_subarray(self, other):
        # focal length choice doesn't matter here, set to equivalent so we don't get
        # an error if only the effective focal length is available in the file
        subarray = SubarrayDescription.from_hdf(
            other, focal_length_choice=FocalLengthKind.EQUIVALENT
        )

        # Check for duplicate telescope IDs when combining telescope events
        if self.combine_telescope_data:
            new_tel_ids = set(subarray.tel_ids)
            duplicates = self.tel_ids_set.intersection(new_tel_ids)
            if duplicates:
                raise ValueError(
                    f"Duplicate telescope IDs found when merging file {other.filename}: {sorted(duplicates)}. "
                    "Each telescope ID must be unique across all input files when using "
                    "the merge strategy 'combine-telescope-data'."
                )
            self.tel_ids_set.update(new_tel_ids)

        self.subarray_list.append(subarray)

        if self.subarray is None:
            self.subarray = subarray
            if not self.combine_telescope_data:
                self.subarray.to_hdf(self.h5file)

        if not self.combine_telescope_data:
            # Relax subarray matching requirements for attaching
            # monitoring data of the same observation block.
            if not self.single_ob or not self.attach_monitoring:
                if self.subarray != subarray:
                    raise CannotMerge(
                        f"Subarrays do not match for file: {other.filename}"
                    )
            else:
                if not SubarrayDescription.check_matching_subarrays(
                    [self.subarray, subarray]
                ):
                    raise CannotMerge(
                        f"Subarrays are not compatible for file: {other.filename}"
                    )

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
