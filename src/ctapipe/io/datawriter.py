#!/usr/bin/env python3
"""
Class to write DL1 (a,b) and DL2 (a) data from an event stream
"""


import pathlib
from collections import defaultdict

import numpy as np
import tables
from traitlets import Dict, Instance

from ..containers import (
    ArrayEventContainer,
    TelescopeConfigurationIndexContainer,
    TelEventIndexContainer,
)
from ..core import Component, Container, Field, Provenance, ToolConfigurationError
from ..core.traits import Bool, CaselessStrEnum, Float, Int, Path, Unicode
from ..instrument import SubarrayDescription
from . import metadata as meta
from .astropy_helpers import write_table
from .datalevels import DataLevel
from .eventsource import EventSource
from .hdf5tableio import HDF5TableWriter
from .tableio import FixedPointColumnTransform, TelListToMaskTransform

__all__ = ["DataWriter", "DATA_MODEL_VERSION", "write_reference_metadata_headers"]

tables.parameters.NODE_CACHE_SLOTS = 3000  # fixes problem with too many datasets


def _get_tel_index(event, tel_id):
    return TelEventIndexContainer(
        obs_id=event.index.obs_id,
        event_id=event.index.event_id,
        tel_id=np.int16(tel_id),
    )


# define the version of the data model written here. This should be updated
# when necessary:
# - increase the major number if there is a breaking change to the model
#   (meaning readers need to update scripts)
# - increase the minor number if new columns or datasets are added
# - increase the patch number if there is a small bugfix to the model.
DATA_MODEL_VERSION = "v6.0.0"
DATA_MODEL_CHANGE_HISTORY = """
- v6.0.0: - Change R1- and DL0-waveform shape from (n_pixels, n_samples) to be always
            (n_channels, n_pixels, n_samples).
- v5.1.0: - Remove redundant 'is_valid' column in ``DispContainer``.
          - Rename content of ``DispContainer`` from 'norm' to 'parameter' and use the same
            default prefix ('disp') for all containers filled by ``DispReconstructor``.
- v5.0.0: - Change DL2 telescope-wise container prefixes from {algorithm}_tel to {algorithm}_tel_{kind}.
            As of now, this only changes 'tel_distance' to 'tel_impact_distance'
- v4.0.0: - Changed how ctapipe-specific metadata is stored in hdf5 attributes.
            This breaks backwards and forwards compatibility for almost everything.
          - Container prefixes are now included for reconstruction algorithms
            and true parameters.
          - Telescope Impact Parameters were added.
          - Effective focal length and nominal focal length are both included
            in the optics description now. Moved ``TelescopeDescription.type``
            to ``OpticsDescription.size_type``. Added ``OpticsDescription.reflector_shape``.
          - n_samples, n_samples_long, n_channels and n_pixels are now part
            of CameraReadout.
          - The reference_location (EarthLocation origin of the telescope coordinates)
            is now included in SubarrayDescription
          - Only unique optics are stored in the optics table
          - include observation configuration
- v3.0.0: reconstructed core uncertainties split in their X-Y components
- v2.2.0: added R0 and R1 outputs
- v2.1.0: hillas and timing parameters are per default saved in telescope frame (degree) as opposed to camera frame (m)
- v2.0.0: Match optics and camera tables using indices instead of names
- v1.2.0: change to more general data model, including also DL2 (DL1 unchanged)
- v1.1.0: images and peak_times can be stored as scaled integers
- v1.0.3: true_image dtype changed from float32 to int32
"""

PROV = Provenance()


def write_reference_metadata_headers(
    obs_ids: list[int],
    subarray: SubarrayDescription,
    writer: "DataWriter",
    is_simulation: bool,
    data_levels,
    contact_info: meta.Contact,
    instrument_info: meta.Instrument,
) -> None:
    """
    Attaches Core Provenence headers to an output HDF5 file.
    Right now this is hard-coded for use with the ctapipe-process tool

    Parameters
    ----------
    output_path: pathlib.Path
        output HDF5 file
    obs_id: int
        observation ID
    subarray:
        SubarrayDescription to get metadata from
    writer: HDF5TableWriter
        output
    data_levels: List[DataLevel]
        list of data levels that were requested/generated
        (e.g. from `DataWriter.datalevels`)
    contact_info: meta.Contact
        contact metadata
    instrument_info: meta.Instrument
        instrument metadata
    """
    activity = PROV.current_activity
    if activity is None and len(PROV.finished_activities) > 0:
        # assume that we write provenance for a "just finished activity"
        activity = PROV.finished_activities[-1]

    activity_meta = meta.Activity.from_provenance(activity.provenance)
    category = "Sim" if is_simulation else "Other"
    reference = meta.Reference(
        contact=contact_info,
        product=meta.Product(
            description="ctapipe Data Product",
            data_category=category,
            data_levels=data_levels,
            data_association="Subarray",
            data_model_name="ASWG",
            data_model_version=DATA_MODEL_VERSION,
            data_model_url="",
            format="hdf5",
        ),
        process=meta.Process(
            type_="Simulation" if is_simulation else "Observation",
            subtype="",
            id_=",".join(str(x) for x in obs_ids),
        ),
        activity=activity_meta,
        instrument=instrument_info,
    )

    if reference.instrument.id_ == "unspecified":
        reference.instrument.id_ = subarray.name

    headers = reference.to_dict()
    meta.write_to_hdf5(headers, writer.h5file)


class DataWriter(Component):
    """
    Serialize a sequence of events into a HDF5 file, in the correct format

    Examples
    --------
    inside a Tool:

    .. code-block:: python

        with DataWriter(parent=self) as write_data:
            for event in source:
                calibrate(event)
                process_images(event)
                write_data(event)
    """

    # pylint: disable=too-many-instance-attributes
    contact_info = Instance(meta.Contact, kw={}).tag(config=True)
    instrument_info = Instance(meta.Instrument, kw={}).tag(config=True)

    context_metadata = Dict(
        help=(
            "Additional metadata keywords and values that describe this data. "
            "This should be a dictionary where the keys will be appended to the "
            "CONTEXT section of the output file's attributes. Keys can be hierarchical "
            "by using a space between each level, e.g. ``SIMULATION PRODUCTION`` "
            "would make a key PRODUCTION grouped under the key SIMULATION"
        )
    ).tag(config=True)

    output_path = Path(
        help="output filename", default_value=pathlib.Path("events.dl1.h5")
    ).tag(config=True)

    write_r0_waveforms = Bool(
        help="Store R0 waveforms if available", default_value=False
    ).tag(config=True)

    write_r1_waveforms = Bool(
        help="Store R1 waveforms if available", default_value=False
    ).tag(config=True)

    write_dl1_images = Bool(
        help="Store DL1 Images if available", default_value=False
    ).tag(config=True)

    write_dl1_parameters = Bool(
        help="Store DL1 image parameters if available", default_value=True
    ).tag(config=True)

    write_dl2 = Bool(
        help="Store DL2 stereo shower parameters if available", default_value=False
    ).tag(config=True)

    write_muon_parameters = Bool(
        help="Store muon parameters if available", default_value=False
    ).tag(config=True)

    compression_level = Int(
        help="compression level, 0=None, 9=maximum", default_value=5, min=0, max=9
    ).tag(config=True)

    compression_type = CaselessStrEnum(
        values=["blosc:zstd", "zlib"],
        help="compressor algorithm to use. ",
        default_value="blosc:zstd",
    ).tag(config=True)

    write_index_tables = Bool(
        help=(
            "Generate PyTables index datasets for all tables that contain an "
            "event_id or tel_id. These speed up in-kernel pytables operations,"
            "but add some overhead to the file. They can also be generated "
            "and attached after the file is written "
        ),
        default_value=False,
    ).tag(config=True)

    overwrite = Bool(help="overwrite output file if it exists").tag(config=True)

    transform_waveform = Bool(default_value=False).tag(config=True)
    waveform_dtype = Unicode(default_value="int32").tag(config=True)
    waveform_offset = Int(default_value=0).tag(config=True)
    waveform_scale = Float(default_value=1000.0).tag(config=True)

    transform_image = Bool(default_value=False).tag(config=True)
    image_dtype = Unicode(default_value="int32").tag(config=True)
    image_offset = Int(default_value=0).tag(config=True)
    image_scale = Float(default_value=10.0).tag(config=True)

    transform_peak_time = Bool(default_value=False).tag(config=True)
    peak_time_dtype = Unicode(default_value="int16").tag(config=True)
    peak_time_offset = Int(default_value=0).tag(config=True)
    peak_time_scale = Float(default_value=100.0).tag(config=True)

    def __init__(self, event_source: EventSource, config=None, parent=None, **kwargs):
        """

        Parameters
        ----------
        event_source : EventSource
            parent event source, which provides header information for the
            subarray, observation, simulation configuration, and the obs_id
        config : , optional
            configuration class
        parent : , optional
            parent of this component in the config hierarchy (this supersedes
            the config option)
        **kwargs :
            other options, such as parameters passed to parent.

        """
        super().__init__(config=config, parent=parent, **kwargs)

        self.event_source = event_source
        self.contact_info = meta.Contact(parent=self)
        self.instrument_info = meta.Instrument(parent=self)

        self._at_least_one_event = False
        self._is_simulation = event_source.is_simulation
        self._subarray: SubarrayDescription = event_source.subarray

        self._hdf5_filters = None

        self._setup_output_path()
        self._setup_compression()
        self._setup_writer()
        self._setup_outputfile()

        # store per-ob for which telescopes we've already written the fixed pointing
        self._constant_telescope_pointing_written = defaultdict(set)

    def _setup_outputfile(self):
        self._subarray.to_hdf(self._writer.h5file)
        self._write_scheduling_and_observation_blocks()
        if self._is_simulation:
            self._write_simulation_configuration()
            self._write_atmosphere_profile(
                "/simulation/service/atmosphere_density_profile"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.finish()

    def __call__(self, event: ArrayEventContainer):
        """
        Write a single event to the output file.
        """
        self._at_least_one_event = True
        self.log.debug("WRITING EVENT %s", event.index)

        self._write_trigger(event)
        # write fixed pointing only for simulation, observed data will have monitoring
        if self._is_simulation:
            self._write_constant_pointing(event)

        if event.simulation is not None and event.simulation.shower is not None:
            self._writer.write(
                table_name="simulation/event/subarray/shower",
                containers=[event.index, event.simulation.shower],
            )

            for tel_id, sim in event.simulation.tel.items():
                table_name = self.table_name(tel_id)
                tel_index = _get_tel_index(event, tel_id)
                self._writer.write(
                    f"simulation/event/telescope/impact/{table_name}",
                    [tel_index, sim.impact],
                )

        if self.write_r1_waveforms:
            self._write_r1_telescope_events(event)

        if self.write_r0_waveforms:
            self._write_r0_telescope_events(event)

        # write telescope event data
        self._write_dl1_telescope_events(event)

        # write DL2 info if requested
        if self.write_dl2:
            self._write_dl2_telescope_events(event)
            self._write_dl2_stereo_event(event)

        if self.write_muon_parameters:
            self._write_muon_telescope_events(event)

    def _write_constant_pointing(self, event):
        """
        Write pointing configuration from event data assuming fixed pointing over the OB.

        This function mainly exists due to a limitation of sim_telarray files.
        Pointing information is only written as part of triggered array events,
        even though it is constant over the run. It also is not written for all
        telescopes, only for those for which it is "known", which seem to be
        all telescopes that participated in an array event at least once.

        So we write the first pointing information for each telescope into the
        configuration table.
        """
        obs_id = event.index.obs_id

        for tel_id, pointing in event.pointing.tel.items():
            if tel_id in self._constant_telescope_pointing_written[obs_id]:
                continue
            index = TelescopeConfigurationIndexContainer(
                obs_id=obs_id,
                tel_id=tel_id,
            )
            self._writer.write(
                f"configuration/telescope/pointing/tel_{tel_id:03d}", (index, pointing)
            )
            self._constant_telescope_pointing_written[obs_id].add(tel_id)

    def finish(self):
        """called after all events are done"""
        self.log.info("Finishing output")
        if not self._at_least_one_event:
            self.log.warning("No events have been written to the output file")

        if self.write_index_tables:
            self._generate_indices()

        write_reference_metadata_headers(
            subarray=self._subarray,
            obs_ids=self.event_source.obs_ids,
            writer=self._writer,
            is_simulation=self._is_simulation,
            data_levels=self.datalevels,
            contact_info=self.contact_info,
            instrument_info=self.instrument_info,
        )

        self._write_context_metadata_headers()
        self._writer.close()
        PROV.add_output_file(str(self.output_path), role="DL1/Event")

    @property
    def datalevels(self):
        """returns a list of data levels requested"""
        data_levels = []
        if self.write_dl1_images:
            data_levels.append(DataLevel.DL1_IMAGES)
        if self.write_dl1_parameters:
            data_levels.append(DataLevel.DL1_PARAMETERS)
        if self.write_muon_parameters:
            data_levels.append(DataLevel.DL1_MUON)
        if self.write_dl2:
            data_levels.append(DataLevel.DL2)
        if self.write_r0_waveforms:
            data_levels.append(DataLevel.R0)
        if self.write_r1_waveforms:
            data_levels.append(DataLevel.R1)
        return data_levels

    def _setup_compression(self):
        """setup HDF5 compression"""
        self._hdf5_filters = tables.Filters(
            complevel=self.compression_level,
            complib=self.compression_type,
            fletcher32=True,  # attach a checksum to each chunk for error correction
        )
        self.log.debug("compression filters: %s", self._hdf5_filters)

    def _setup_output_path(self):
        """
        ensure output path exists, and if requested delete what is there for
        overwriting
        """
        self.output_path = self.output_path.expanduser()
        if self.output_path.exists():
            if self.overwrite:
                self.log.warning("Overwriting %s", self.output_path)
                self.output_path.unlink()
            else:
                raise ToolConfigurationError(
                    f"Output file {self.output_path} exists"
                    ", use the `overwrite` option or choose another `output_path` "
                )
        self.log.debug("output path: %s", self.output_path)

        # check that options make sense
        writable_things = [
            self.write_dl1_parameters,
            self.write_dl1_images,
            self.write_dl2,
            self.write_r1_waveforms,
            self.write_muon_parameters,
        ]
        if not any(writable_things):
            self.log.warning(
                "No processing results were selected for writing"
                ", only writing trigger and simulation information"
            )

    def _setup_writer(self):
        """
        Create a TableWriter and setup any column exclusions
        When complete, self._writer should be initialized
        """
        writer = HDF5TableWriter(
            str(self.output_path),
            parent=self,
            mode="a",
            add_prefix=True,
            filters=self._hdf5_filters,
        )

        tr_tel_list_to_mask = TelListToMaskTransform(self._subarray)

        writer.add_column_transform(
            table_name="dl1/event/subarray/trigger",
            col_name="tels_with_trigger",
            transform=tr_tel_list_to_mask,
        )

        writer.exclude("/dl1/event/telescope/trigger", "trigger_pixels")
        writer.exclude("/dl1/event/telescope/images/.*", "parameters")
        writer.exclude("/simulation/event/telescope/images/.*", "true_parameters")

        if not self.write_dl1_images:
            writer.exclude("/simulation/event/telescope/images/.*", "true_image")

        if not self.write_dl1_parameters:
            writer.exclude("/dl1/event/telescope/images/.*", "image_mask")

        # Set up transforms
        if self.transform_image:
            transform = FixedPointColumnTransform(
                scale=self.image_scale,
                offset=self.image_offset,
                source_dtype=np.float32,
                target_dtype=np.dtype(self.image_dtype),
            )
            writer.add_column_transform_regexp(
                "dl1/event/telescope/images/.*", "image", transform
            )

        if self.transform_waveform:
            transform = FixedPointColumnTransform(
                scale=self.waveform_scale,
                offset=self.waveform_offset,
                source_dtype=np.float32,
                target_dtype=np.dtype(self.waveform_dtype),
            )
            writer.add_column_transform_regexp(
                "r1/event/telescope/.*", "waveform", transform
            )

        if self.transform_peak_time:
            transform = FixedPointColumnTransform(
                scale=self.peak_time_scale,
                offset=self.peak_time_offset,
                source_dtype=np.float32,
                target_dtype=np.dtype(self.peak_time_dtype),
            )
            writer.add_column_transform_regexp(
                "dl1/event/telescope/images/.*", "peak_time", transform
            )

        # set up DL2 transforms:
        # - the single-tel output has no list of telescopes
        # - the stereo output telescope list needs to be transformed to a pattern
        writer.exclude("dl2/event/telescope/.*", ".*telescopes")
        writer.add_column_transform_regexp(
            table_regexp="dl2/event/subarray/.*",
            col_regexp=".*telescopes",
            transform=tr_tel_list_to_mask,
        )

        # final initialization
        self._writer = writer
        self.log.debug("Writer initialized: %s", self._writer)

    def _write_scheduling_and_observation_blocks(self):
        """write out SB and OB info"""

        self.log.debug(
            "writing %d sbs and %d obs",
            len(self.event_source.scheduling_blocks.values()),
            len(self.event_source.observation_blocks.values()),
        )

        for sb in self.event_source.scheduling_blocks.values():
            self._writer.write("configuration/observation/scheduling_block", sb)

        for ob in self.event_source.observation_blocks.values():
            self._writer.write("configuration/observation/observation_block", ob)

    def _write_simulation_configuration(self):
        """
        Write the simulation headers to a single row of a table. Later
        if this file is merged with others, that table will grow.

        Note that this function should be run first
        """
        self.log.debug("Writing simulation configuration")

        class ExtraSimInfo(Container):
            """just to contain obs_id"""

            default_prefix = ""
            obs_id = Field(0, "Simulation Run Identifier")

        for obs_id, config in self.event_source.simulation_config.items():
            extramc = ExtraSimInfo(obs_id=obs_id)
            config.prefix = ""

            self._writer.write("configuration/simulation/run", [extramc, config])

    def write_simulated_shower_distributions(self, distributions):
        """Write the distribution of thrown showers."""

        self.log.debug("Writing simulation histograms")

        for container in distributions.values():
            self._writer.write(
                table_name="simulation/service/shower_distribution",
                containers=container,
            )

    def table_name(self, tel_id):
        """construct dataset table names depending on chosen split method"""
        return f"tel_{tel_id:03d}"

    def _write_trigger(self, event: ArrayEventContainer):
        """
        Write trigger information
        """
        self._writer.write(
            table_name="dl1/event/subarray/trigger",
            containers=[event.index, event.trigger],
        )

        for tel_id, trigger in event.trigger.tel.items():
            self._writer.write(
                "dl1/event/telescope/trigger", (_get_tel_index(event, tel_id), trigger)
            )

    def _write_r1_telescope_events(self, event: ArrayEventContainer):
        for tel_id, r1_tel in event.r1.tel.items():
            tel_index = _get_tel_index(event, tel_id)
            table_name = self.table_name(tel_id)

            r1_tel.prefix = ""
            self._writer.write(f"r1/event/telescope/{table_name}", [tel_index, r1_tel])

    def _write_r0_telescope_events(self, event: ArrayEventContainer):
        for tel_id, r0_tel in event.r0.tel.items():
            tel_index = _get_tel_index(event, tel_id)
            table_name = self.table_name(tel_id)

            r0_tel.prefix = ""
            self._writer.write(f"r0/event/telescope/{table_name}", [tel_index, r0_tel])

    def _write_dl1_telescope_events(self, event: ArrayEventContainer):
        """
        add entries to the event/telescope tables for each telescope in a single
        event
        """

        for tel_id, dl1_camera in event.dl1.tel.items():
            tel_index = _get_tel_index(event, tel_id)

            dl1_camera.prefix = ""  # don't want a prefix for this container
            telescope = self._subarray.tel[tel_id]
            self.log.debug("WRITING TELESCOPE %s: %s", tel_id, telescope)

            table_name = self.table_name(tel_id)

            if self.write_dl1_parameters:
                self._writer.write(
                    table_name=f"dl1/event/telescope/parameters/{table_name}",
                    containers=[tel_index, *dl1_camera.parameters.values()],
                )

            if self.write_dl1_images:
                if dl1_camera.image is None:
                    raise ValueError(
                        "DataWriter.write_dl1_images is True but event does not contain image"
                    )

                self._writer.write(
                    table_name=f"dl1/event/telescope/images/{table_name}",
                    containers=[tel_index, dl1_camera],
                )

            if self._is_simulation:
                # always write this, so that at least the sum is included
                self._writer.write(
                    f"simulation/event/telescope/images/{table_name}",
                    [tel_index, event.simulation.tel[tel_id]],
                )

                has_sim_image = (
                    tel_id in event.simulation.tel
                    and event.simulation.tel[tel_id].true_image is not None
                )
                if self.write_dl1_parameters and has_sim_image:
                    true_parameters = event.simulation.tel[tel_id].true_parameters
                    # only write the available containers, no peak time related
                    # features for true image available.
                    self._writer.write(
                        f"simulation/event/telescope/parameters/{table_name}",
                        [
                            tel_index,
                            true_parameters.hillas,
                            true_parameters.leakage,
                            true_parameters.concentration,
                            true_parameters.morphology,
                            true_parameters.intensity_statistics,
                        ],
                    )

    def _write_muon_telescope_events(self, event: ArrayEventContainer):
        for tel_id, muon in event.muon.tel.items():
            table_name = self.table_name(tel_id)
            tel_index = _get_tel_index(event, tel_id)
            self._writer.write(
                f"dl1/event/telescope/muon/{table_name}",
                [tel_index, muon.ring, muon.parameters, muon.efficiency],
            )

    def _write_dl2_telescope_events(self, event: ArrayEventContainer):
        """
        write per-telescope DL2 shower information.

        Currently this writes to a single table per type of shower
        reconstruction and per algorithm, with all telescopes combined.
        """

        for tel_id, dl2_tel in event.dl2.tel.items():
            table_name = self.table_name(tel_id)

            tel_index = _get_tel_index(event, tel_id)
            for container_name, algorithm_map in dl2_tel.items():
                for algorithm, container in algorithm_map.items():
                    name = (
                        f"dl2/event/telescope/{container_name}/{algorithm}/{table_name}"
                    )

                    self._writer.write(
                        table_name=name, containers=[tel_index, container]
                    )

    def _write_dl2_stereo_event(self, event: ArrayEventContainer):
        """
        write per-telescope DL2 shower information to e.g.
        `/dl2/event/stereo/{geometry,energy,classification}/<algorithm_name>`
        """
        # pylint: disable=no-self-use
        for container_name, algorithm_map in event.dl2.stereo.items():
            for algorithm, container in algorithm_map.items():
                # note this will only write info if the particular algorithm
                # generated it (otherwise the algorithm map is empty, and no
                # data will be written)
                self._writer.write(
                    table_name=f"dl2/event/subarray/{container_name}/{algorithm}",
                    containers=[event.index, container],
                )

    def _generate_table_indices(self, h5file, start_node):
        """helper to generate PyTables index tabnles for common columns"""
        for node in h5file.iter_nodes(start_node):
            if not isinstance(node, tables.group.Group):
                self.log.debug("generating indices for node: %s", node)
                if "event_id" in node.colnames:
                    node.cols.event_id.create_index()
                    self.log.debug("generated event_id index")
                if "tel_id" in node.colnames:
                    node.cols.tel_id.create_index()
                    self.log.debug("generated tel_id index")
                if "obs_id" in node.colnames:
                    self.log.debug("generated obs_id index")
                    node.cols.obs_id.create_index(kind="ultralight")
            else:
                # recurse
                self._generate_table_indices(h5file, node)

    def _generate_indices(self):
        """generate PyTables index tables for common columns"""
        self.log.debug("Writing index tables")
        if self.write_dl1_images:
            self._generate_table_indices(
                self._writer.h5file, "/dl1/event/telescope/images"
            )
            if self._is_simulation:
                self._generate_table_indices(
                    self._writer.h5file, "/simulation/event/telescope/images"
                )
        if self.write_dl1_parameters:
            self._generate_table_indices(
                self._writer.h5file, "/dl1/event/telescope/parameters"
            )
            if self._is_simulation:
                self._generate_table_indices(
                    self._writer.h5file, "/simulation/event/telescope/parameters"
                )

        self._generate_table_indices(self._writer.h5file, "/dl1/event/subarray")

    def _write_context_metadata_headers(self):
        """write out any user-defined metadata in the context_metadata field to the
        headers.

        This will create a set of headers that start with CONTEXT <KEY> = value

        Keys can be hierarchical separated by spaces

        """

        # first append CONTEXT to each key in the dict

        context_dict = {}

        for key, value in self.context_metadata.items():
            key = " ".join(["CONTEXT", key])
            context_dict[key] = value

        meta.write_to_hdf5(context_dict, self._writer.h5file)

    def _write_atmosphere_profile(self, path):
        """
        write atmosphere profiles if they are in a tabular format

        Parameters
        ----------
        path: str
            path in the HDF5 file where to place the profile

        """

        profile = self.event_source.atmosphere_density_profile

        if profile:
            if hasattr(profile, "table"):
                write_table(
                    table=profile.table,
                    h5file=self._writer.h5file,
                    path=path,
                    append=False,
                )
            else:
                self.logger.warning(
                    f"The AtmosphereDensityProfile type '{profile.__class__.__name__}' "
                    "is not serializable. No atmosphere profile will be stored in the "
                    "output file"
                )
