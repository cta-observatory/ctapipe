#!/usr/bin/env python3
"""
Class to write DL1 (a,b) and DL2 (a) data from an event stream
"""


import pathlib
from collections import defaultdict
from traitlets import Instance

import numpy as np
import tables
from astropy import units as u

from ..containers import (
    ArrayEventContainer,
    SimulatedShowerDistribution,
    TelEventIndexContainer,
)
from ..core import Component, Container, Field, Provenance, ToolConfigurationError
from ..core.traits import Bool, CaselessStrEnum, Float, Int, Path, Unicode
from ..instrument import SubarrayDescription
from . import EventSource, HDF5TableWriter, TableWriter
from . import metadata as meta
from .datalevels import DataLevel
from .simteleventsource import SimTelEventSource
from .tableio import FixedPointColumnTransform, TelListToMaskTransform

__all__ = ["DataWriter", "DATA_MODEL_VERSION", "write_reference_metadata_headers"]

tables.parameters.NODE_CACHE_SLOTS = 3000  # fixes problem with too many datasets


def _get_tel_index(event, tel_id):
    return TelEventIndexContainer(
        obs_id=event.index.obs_id,
        event_id=event.index.event_id,
        tel_id=np.int16(tel_id),
    )


# define the version of the DL1 data model written here. This should be updated
# when necessary:
# - increase the major number if there is a breaking change to the model
#   (meaning readers need to update scripts)
# - increase the minor number if new columns or datasets are added
# - increase the patch number if there is a small bugfix to the model.
DATA_MODEL_VERSION = "v3.0.0"
DATA_MODEL_CHANGE_HISTORY = """
- v3.0.0: reconstructed core uncertainties splitted in their X-Y components
- v2.2.0: added R0 and R1 outputs
- v2.1.0: hillas and timing parameters are per default saved in telescope frame (degree) as opposed to camera frame (m)
- v2.0.0: Match optics and camera tables using indices instead of names
- v1.2.0: change to more general data model, including also DL2 (DL1 unchanged)
- v1.1.0: images and peak_times can be stored as scaled integers
- v1.0.3: true_image dtype changed from float32 to int32
"""

PROV = Provenance()


def write_reference_metadata_headers(
    obs_ids, subarray, writer, is_simulation, data_levels, contact_info
):
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
    """
    activity = PROV.current_activity.provenance
    category = "Sim" if is_simulation else "Other"

    reference = meta.Reference(
        contact=contact_info,
        product=meta.Product(
            description="ctapipe Data Product",
            data_category=category,
            data_level=[l.name for l in data_levels],
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
        activity=meta.Activity.from_provenance(activity),
        instrument=meta.Instrument(
            site="Other",  # need a way to detect site...
            class_="Subarray",
            type_="unknown",
            version="unknown",
            id_=subarray.name,
        ),
    )

    headers = reference.to_dict()
    meta.write_to_hdf5(headers, writer.h5file)


class DataWriter(Component):
    """
    Serialize a sequence of events into a HDF5 DL1 file, in the correct format

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

    output_path = Path(
        help="output filename", default_value=pathlib.Path("events.dl1.h5")
    ).tag(config=True)

    write_raw_waveforms = Bool(
        help="Store R0 waveforms if available", default_value=False
    ).tag(config=True)

    write_waveforms = Bool(
        help="Store R1 waveforms if available", default_value=False
    ).tag(config=True)

    write_images = Bool(help="Store DL1 Images if available", default_value=False).tag(
        config=True
    )

    write_parameters = Bool(
        help="Store DL1 image parameters if available", default_value=True
    ).tag(config=True)

    write_stereo_shower = Bool(
        help="Store DL2 stereo shower parameters if available", default_value=False
    ).tag(config=True)

    write_mono_shower = Bool(
        help="Store DL2 mono parameters if available", default_value=False
    ).tag(config=True)

    compression_level = Int(
        help="compression level, 0=None, 9=maximum", default_value=5, min=0, max=9
    ).tag(config=True)

    split_datasets_by = CaselessStrEnum(
        values=["tel_id", "tel_type"],
        default_value="tel_id",
        help="Splitting level for the DL1 parameters and images datasets",
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
            parent of this component in the config hierarchy (this supercedes
            the config option)
        **kwargs :
            other options, such as parameters passed to parent.

        """
        super().__init__(config=config, parent=parent, **kwargs)

        self.event_source = event_source
        self.contact_info = meta.Contact(parent=self)

        self._at_least_one_event = False
        self._is_simulation = event_source.is_simulation
        self._subarray: SubarrayDescription = event_source.subarray

        self._hdf5_filters = None
        self._writer: HDF5TableWriter = None

        self._setup_output_path()
        self._setup_compression()
        self._setup_writer()
        self._setup_outputfile()

        # store last pointing to only write unique poitings
        self._last_pointing = None
        self._last_pointing_tel = defaultdict(lambda: (np.nan * u.deg, np.nan * u.deg))

    def _setup_outputfile(self):
        self._subarray.to_hdf(self._writer.h5file)
        if self._is_simulation:
            self._write_simulation_configuration()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.finish()

    def __call__(self, event: ArrayEventContainer):
        """
        Write a single event to the output file.
        """
        self._at_least_one_event = True

        # Write subarray event data
        self._write_subarray_pointing(event, writer=self._writer)

        self.log.debug("WRITING EVENT %s", event.index)
        self._writer.write(
            table_name="dl1/event/subarray/trigger",
            containers=[event.index, event.trigger],
        )
        if self._is_simulation:
            self._writer.write(
                table_name="simulation/event/subarray/shower",
                containers=[event.index, event.simulation.shower],
            )

        if self.write_waveforms:
            self._write_r1_telescope_events(self._writer, event)

        if self.write_raw_waveforms:
            self._write_r0_telescope_events(self._writer, event)

        # write telescope event data
        self._write_dl1_telescope_events(self._writer, event)

        # write DL2 info if requested
        if self.write_mono_shower:
            self._write_dl2_telescope_events(self._writer, event)

        if self.write_stereo_shower:
            self._write_dl2_stereo_event(self._writer, event)

    def finish(self):
        """called after all events are done"""
        self.log.info("Finishing DL1 output")
        if not self._at_least_one_event:
            self.log.warning("No events have been written to the output file")
        if self._writer:
            if self.write_index_tables:
                self._generate_indices()

            write_reference_metadata_headers(
                subarray=self._subarray,
                obs_ids=self.event_source.obs_ids,
                writer=self._writer,
                is_simulation=self._is_simulation,
                data_levels=self.datalevels,
                contact_info=self.contact_info,
            )
            self._writer.close()
            self._writer = None

    @property
    def datalevels(self):
        """returns a list of data levels requested"""
        data_levels = []
        if self.write_images:
            data_levels.append(DataLevel.DL1_IMAGES)
        if self.write_parameters:
            data_levels.append(DataLevel.DL1_PARAMETERS)
        if self.write_stereo_shower or self.write_mono_shower:
            data_levels.append(DataLevel.DL2)
        if self.write_raw_waveforms:
            data_levels.append(DataLevel.R0)
        if self.write_waveforms:
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
        PROV.add_output_file(str(self.output_path), role="DL1/Event")

        # check that options make sense
        writable_things = [
            self.write_parameters,
            self.write_images,
            self.write_mono_shower,
            self.write_stereo_shower,
            self.write_waveforms,
            self.write_parameters,
        ]
        if not any(writable_things):
            raise ToolConfigurationError(
                "DataWriter configured to write no information"
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

        # avoid some warnings about unwritable columns (which here are just
        # sub-containers)
        writer.exclude("dl1/event/subarray/trigger", "tel")
        writer.exclude("dl1/monitoring/subarray/pointing", "tel")
        writer.exclude("/dl1/event/telescope/images/.*", "parameters")

        # currently the trigger info is used for the event time, but we dont'
        # want the other bits of the trigger container in the pointing or other
        # montitoring containers
        writer.exclude("dl1/monitoring/subarray/pointing", "event_type")
        writer.exclude("dl1/monitoring/subarray/pointing", "tels_with_trigger")
        writer.exclude("dl1/monitoring/subarray/pointing", "n_trigger_pixels")
        writer.exclude("/dl1/event/telescope/trigger", "trigger_pixels")
        writer.exclude("/dl1/monitoring/telescope/pointing/.*", "n_trigger_pixels")
        writer.exclude("/dl1/monitoring/telescope/pointing/.*", "trigger_pixels")
        writer.exclude("/dl1/monitoring/event/pointing/.*", "event_type")

        if self.write_parameters is False:
            writer.exclude("/dl1/event/telescope/images/.*", "image_mask")

        if self._is_simulation:
            writer.exclude("/simulation/event/telescope/images/.*", "true_parameters")
            # no timing information yet for true images
            writer.exclude("/simulation/event/telescope/parameters/.*", r"peak_time_.*")
            writer.exclude("/simulation/event/telescope/parameters/.*", "timing_.*")
            writer.exclude("/simulation/event/subarray/shower", "true_tel")

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
        # - the single-tel output has no list of tel_ids
        # - the stereo output tel_ids list needs to be transformed to a pattern
        writer.exclude("dl2/event/telescope/.*", "tel_ids")
        writer.add_column_transform_regexp(
            table_regexp="dl2/event/subarray/.*",
            col_regexp="tel_ids",
            transform=tr_tel_list_to_mask,
        )

        # final initialization
        self._writer = writer
        self.log.debug("Writer initialized: %s", self._writer)

    def _write_subarray_pointing(self, event: ArrayEventContainer, writer: TableWriter):
        """store subarray pointing info in a monitoring table"""
        pnt = event.pointing
        current_pointing = (pnt.array_azimuth, pnt.array_altitude)
        if current_pointing != self._last_pointing:
            pnt.prefix = ""
            writer.write("dl1/monitoring/subarray/pointing", [event.trigger, pnt])
            self._last_pointing = current_pointing

    def _write_simulation_configuration(self):
        """
        Write the simulation headers to a single row of a table. Later
        if this file is merged with others, that table will grow.

        Note that this function should be run first
        """
        self.log.debug("Writing simulation configuration")

        class ExtraSimInfo(Container):
            """just to contain obs_id"""

            container_prefix = ""
            obs_id = Field(0, "Simulation Run Identifier")

        for obs_id, config in self.event_source.simulation_config.items():
            extramc = ExtraSimInfo(obs_id=obs_id)
            config.prefix = ""

            self._writer.write("configuration/simulation/run", [extramc, config])

    def write_simulation_histograms(self, event_source):
        """Write the distribution of thrown showers

        Notes
        -----
        - this only runs if this is a simulation file. The current
          implementation is a bit of a hack and implies we should improve
          SimTelEventSource to read this info.

        - Currently the histograms are at the end of the simtel file, so if
          max_events is set to non-zero, the end of the file may not be read,
          and this no histograms will be found.
        """
        if not self._is_simulation:
            self.log.debug("Not writing simulation histograms for observed data")
            return

        if not isinstance(event_source, SimTelEventSource):
            self.log.debug("Not writing simulation for non-SimTelEventSource")
            return

        self.log.debug("Writing simulation histograms")

        def fill_from_simtel(
            obs_id, eventio_hist, container: SimulatedShowerDistribution
        ):
            """fill from a SimTel Histogram entry"""
            container.obs_id = obs_id
            container.hist_id = eventio_hist["id"]
            container.num_entries = eventio_hist["entries"]
            xbins = np.linspace(
                eventio_hist["lower_x"],
                eventio_hist["upper_x"],
                eventio_hist["n_bins_x"] + 1,
            )
            ybins = np.linspace(
                eventio_hist["lower_y"],
                eventio_hist["upper_y"],
                eventio_hist["n_bins_y"] + 1,
            )

            container.bins_core_dist = xbins * u.m
            container.bins_energy = 10**ybins * u.TeV
            container.histogram = eventio_hist["data"]
            container.meta["hist_title"] = eventio_hist["title"]
            container.meta["x_label"] = "Log10 E (TeV)"
            container.meta["y_label"] = "3D Core Distance (m)"

        hists = event_source.file_.histograms
        if hists is not None:
            hist_container = SimulatedShowerDistribution()
            hist_container.prefix = ""
            for hist in hists:
                if hist["id"] == 6:
                    fill_from_simtel(self.event_source.obs_ids[0], hist, hist_container)
                    self._writer.write(
                        table_name="simulation/service/shower_distribution",
                        containers=hist_container,
                    )

    def table_name(self, tel_id, tel_type):
        """construct dataset table names depending on chosen split method"""
        return f"tel_{tel_id:03d}" if self.split_datasets_by == "tel_id" else tel_type

    def _write_r1_telescope_events(
        self, writer: TableWriter, event: ArrayEventContainer
    ):
        for tel_id, r1_tel in event.r1.tel.items():

            tel_index = TelEventIndexContainer(
                obs_id=event.index.obs_id,
                event_id=event.index.event_id,
                tel_id=np.int16(tel_id),
            )
            telescope = self._subarray.tel[tel_id]
            table_name = self.table_name(tel_id, str(telescope))

            r1_tel.prefix = ""
            writer.write(f"r1/event/telescope/{table_name}", [tel_index, r1_tel])

    def _write_r0_telescope_events(
        self, writer: TableWriter, event: ArrayEventContainer
    ):
        for tel_id, r0_tel in event.r0.tel.items():

            tel_index = TelEventIndexContainer(
                obs_id=event.index.obs_id,
                event_id=event.index.event_id,
                tel_id=np.int16(tel_id),
            )
            telescope = self._subarray.tel[tel_id]
            table_name = self.table_name(tel_id, str(telescope))

            r0_tel.prefix = ""
            writer.write(f"r0/event/telescope/{table_name}", [tel_index, r0_tel])

    def _write_dl1_telescope_events(
        self, writer: TableWriter, event: ArrayEventContainer
    ):
        """
        add entries to the event/telescope tables for each telescope in a single
        event
        """

        # write the telescope tables

        # pointing info
        for tel_id, pnt in event.pointing.tel.items():
            current_pointing = (pnt.azimuth, pnt.altitude)
            if current_pointing != self._last_pointing_tel[tel_id]:
                pnt.prefix = ""
                writer.write(
                    f"dl1/monitoring/telescope/pointing/tel_{tel_id:03d}",
                    [event.trigger.tel[tel_id], pnt],
                )
                self._last_pointing_tel[tel_id] = current_pointing

        # trigger info
        for tel_id, trigger in event.trigger.tel.items():
            writer.write(
                "dl1/event/telescope/trigger", [_get_tel_index(event, tel_id), trigger]
            )

        for tel_id, dl1_camera in event.dl1.tel.items():
            tel_index = _get_tel_index(event, tel_id)

            dl1_camera.prefix = ""  # don't want a prefix for this container
            telescope = self._subarray.tel[tel_id]
            self.log.debug("WRITING TELESCOPE %s: %s", tel_id, telescope)

            table_name = self.table_name(tel_id, str(telescope))

            has_sim_camera = self._is_simulation and (
                tel_id in event.simulation.tel
                and event.simulation.tel[tel_id].true_image is not None
            )

            if self.write_parameters:
                writer.write(
                    table_name=f"dl1/event/telescope/parameters/{table_name}",
                    containers=[tel_index, *dl1_camera.parameters.values()],
                )

                if has_sim_camera:
                    writer.write(
                        f"simulation/event/telescope/parameters/{table_name}",
                        [
                            tel_index,
                            *event.simulation.tel[tel_id].true_parameters.values(),
                        ],
                    )

            if self.write_images:
                if dl1_camera.image is None:
                    raise ValueError(
                        "DataWriter.write_images is True but event does not contain image"
                    )
                # note that we always write the image, even if the image quality
                # criteria are not met (those are only to determine if the parameters
                # can be computed).
                self.log.debug("WRITING IMAGES")
                writer.write(
                    table_name=f"dl1/event/telescope/images/{table_name}",
                    containers=[tel_index, dl1_camera],
                )

                if self._is_simulation and has_sim_camera:
                    writer.write(
                        f"simulation/event/telescope/images/{table_name}",
                        [tel_index, event.simulation.tel[tel_id]],
                    )

    def _write_dl2_telescope_events(
        self, writer: TableWriter, event: ArrayEventContainer
    ):
        """
        write per-telescope DL2 shower information.

        Currently this writes to a single table per type of shower
        reconstruction and per algorithm, with all telescopes combined.
        """

        for tel_id, dl2_tel in event.dl2.tel.items():

            telescope = self._subarray.tel[tel_id]
            table_name = self.table_name(tel_id, str(telescope))

            tel_index = TelEventIndexContainer(
                obs_id=event.index.obs_id,
                event_id=event.index.event_id,
                tel_id=np.int16(tel_id),
            )

            for container_name, algorithm_map in dl2_tel.items():
                for algorithm, container in algorithm_map.items():
                    name = (
                        f"dl2/event/telescope/{container_name}/{algorithm}/{table_name}"
                    )

                    writer.write(table_name=name, containers=[tel_index, container])

    def _write_dl2_stereo_event(self, writer: TableWriter, event: ArrayEventContainer):
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
                writer.write(
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
        if self.write_images:
            self._generate_table_indices(
                self._writer.h5file, "/dl1/event/telescope/images"
            )
            if self._is_simulation:
                self._generate_table_indices(
                    self._writer.h5file, "/simulation/event/telescope/images"
                )
        if self.write_parameters:
            self._generate_table_indices(
                self._writer.h5file, "/dl1/event/telescope/parameters"
            )
            if self._is_simulation:
                self._generate_table_indices(
                    self._writer.h5file, "/simulation/event/telescope/parameters"
                )

        self._generate_table_indices(self._writer.h5file, "/dl1/event/subarray")
