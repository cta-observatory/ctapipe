#!/usr/bin/env python3
"""
Class to write DL1 data from an event stream
"""


import pathlib
from collections import defaultdict
from typing import DefaultDict, Tuple

import numpy as np
import tables
from astropy import units as u

from ..containers import (
    ArrayEventContainer,
    SimulatedShowerDistribution,
    TelEventIndexContainer,
)
from ..core import Component, Container, Field, Provenance, ToolConfigurationError
from ..core.traits import Bool, CaselessStrEnum, Int, Path
from ..io import EventSource, HDF5TableWriter, TableWriter
from ..io.simteleventsource import SimTelEventSource
from ..io import metadata as meta
from ..instrument import SubarrayDescription

__all__ = ["DL1Writer", "DL1_DATA_MODEL_VERSION", "write_reference_metadata_headers"]

tables.parameters.NODE_CACHE_SLOTS = 3000  # fixes problem with too many datasets

DL1_DATA_MODEL_VERSION = "v1.0.2"
PROV = Provenance()


def write_reference_metadata_headers(obs_id, subarray, writer):
    """
    Attaches Core Provenence headers to an output HDF5 file.
    Right now this is hard-coded for use with the ctapipe-stage1-process tool

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
    """
    activity = PROV.current_activity.provenance

    reference = meta.Reference(
        contact=meta.Contact(name="", email="", organization="CTA Consortium"),
        product=meta.Product(
            description="DL1 Data Product",
            data_category="S",
            data_level="DL1",
            data_association="Subarray",
            data_model_name="ASWG DL1",
            data_model_version=DL1_DATA_MODEL_VERSION,
            data_model_url="",
            format="hdf5",
        ),
        process=meta.Process(type_="Simulation", subtype="", id_=int(obs_id)),
        activity=meta.Activity.from_provenance(activity),
        instrument=meta.Instrument(
            site="Other",  # need a way to detect site...
            class_="Subarray",
            type_="unknown",
            version="unknown",
            id_=subarray.name,
        ),
    )

    # TODO: add activity_stop_time?
    headers = reference.to_dict()
    meta.write_to_hdf5(headers, writer._h5file)


class DL1Writer(Component):
    """
    Serialize a sequence of events into a HDF5 DL1 file, in the correct format

    Examples
    --------
    inside a Tool:

    .. code-block:: python

        with DL1Writer(parent=self) as write_dl1:
            for event in source:
                calibrate(event)
                process_images(event)
                write_dl1(event)
    """

    output_path = Path(
        help="DL1 output filename", default_value=pathlib.Path("events.dl1.h5")
    ).tag(config=True)

    write_images = Bool(
        help="Store DL1/Event/Image data in output", default_value=False
    ).tag(config=True)

    write_parameters = Bool(
        help="Compute and store image parameters", default_value=True
    ).tag(config=True)

    compression_level = Int(
        help="compression level, 0=None, 9=maximum", default_value=5, min=0, max=9
    ).tag(config=True)

    split_datasets_by = CaselessStrEnum(
        values=["tel_id", "tel_type"],
        default_value="tel_id",
        help="Splitting level for the parameters and images datasets",
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

        # here we just set up data, but all real initializtion should be in
        # setup(), which is called when the first event is read.

        self._is_simulation = event_source.is_simulation
        self._subarray: SubarrayDescription = event_source.subarray

        if self._is_simulation:
            self._simulation_config = event_source.simulation_config
        else:
            self._simulation_config = None

        self._obs_id = event_source.obs_ids[0]
        self._hdf5_filters = None
        self._last_pointing_tel: DefaultDict[Tuple] = None
        self._last_pointing: Tuple = None
        self._writer: TableWriter = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.finish()

    def __call__(self, event: ArrayEventContainer):
        """
        Write a single event to the output file. On the first event, the output
        file is set up
        """

        # perform delayed initialization on first event
        if self._writer is None:
            self.setup()

        # Write subarray evvent data
        self._write_subarray_pointing(event, writer=self._writer)

        self.log.debug(f"WRITING EVENT {event.index}")
        self._writer.write(
            table_name="dl1/event/subarray/trigger",
            containers=[event.index, event.trigger],
        )
        if self._is_simulation:
            self._writer.write(
                table_name="simulation/event/subarray/shower",
                containers=[event.index, event.simulation.shower],
            )

        # write telescope event data
        self._write_telescope_events(self._writer, event)

    def setup(self):
        """called on first event"""
        self.log.debug("Setting Up DL1 Output")

        self._setup_output_path()
        self._subarray.to_hdf(self.output_path)  # must be first (uses astropy io)
        self._setup_compression()
        self._setup_writer()
        if self._is_simulation:
            self._write_simulation_configuration()

        # store last pointing to only write unique poitings
        self._last_pointing_tel = defaultdict(lambda: (np.nan * u.deg, np.nan * u.deg))

    def finish(self):
        """ called after all events are done """
        self.log.info("Finishing DL1 output")
        if self._writer:
            if self.write_index_tables:
                self._generate_indices()
            write_reference_metadata_headers(
                subarray=self._subarray, obs_id=self._obs_id, writer=self._writer
            )
            self._writer.close()
            self._writer = None

    def _setup_compression(self):
        """ setup HDF5 compression"""
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
                self.log.warning(f"Overwriting {self.output_path}")
                self.output_path.unlink()
            else:
                raise ToolConfigurationError(
                    f"Output file {self.output_path} exists"
                    ", use the `overwrite` option or choose another `output_path` "
                )
        self.log.debug("output path: %s", self.output_path)
        PROV.add_output_file(str(self.output_path), role="DL1/Event")

        # check that options make sense
        if self.write_parameters is False and self.write_images is False:
            raise ToolConfigurationError(
                "The options 'write_parameters' and 'write_images' are "
                "both set to False. No output will be generated in that case. "
                "Please enable one or both of these options."
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

        writer.add_column_transform(
            table_name="dl1/event/subarray/trigger",
            col_name="tels_with_trigger",
            transform=self._subarray.tel_ids_to_mask,
        )

        # exclude some columns that are not writable
        writer.exclude("dl1/event/subarray/trigger", "tel")
        writer.exclude("dl1/monitoring/subarray/pointing", "tel")
        writer.exclude("dl1/monitoring/subarray/pointing", "event_type")
        writer.exclude("dl1/monitoring/subarray/pointing", "tels_with_trigger")
        writer.exclude("dl1/monitoring/subarray/pointing", "n_trigger_pixels")
        writer.exclude("/dl1/event/telescope/trigger", "trigger_pixels")

        table_names_tel_id = [f"tel_{tel_id:03d}" for tel_id in self._subarray.tel]
        table_names_tel_type = [
            str(telescope) for telescope in self._subarray.telescope_types
        ]

        if self.split_datasets_by == "tel_id":
            table_names = table_names_tel_id
        elif self.split_datasets_by == "tel_type":
            table_names = table_names_tel_type
        else:
            table_names = []

        for table_name in table_names:
            if self.write_parameters is False:
                writer.exclude(
                    f"/dl1/event/telescope/images/{table_name}", "image_mask"
                )

            tel_pointing = f"/dl1/monitoring/telescope/pointing/{table_name}"
            writer.exclude(tel_pointing, "n_trigger_pixels")
            writer.exclude(tel_pointing, "trigger_pixels")
            writer.exclude(f"/dl1/event/telescope/images/{table_name}", "parameters")

            if self._is_simulation:
                writer.exclude(
                    f"/simulation/event/telescope/images/{table_name}",
                    "true_parameters",
                )
                # no timing information yet for true images
                writer.exclude(
                    f"/simulation/event/telescope/parameters/{table_name}",
                    r"peak_time_.*",
                )
                writer.exclude(
                    f"/simulation/event/telescope/parameters/{table_name}", r"timing_.*"
                )
                writer.exclude("/simulation/event/subarray/shower", "true_tel")

        for table_name in table_names_tel_id:
            writer.exclude(f"/dl1/monitoring/event/pointing/{table_name}", "event_type")

        self._writer = writer
        self.log.debug("Writer initialized: %s", self._writer)

    def _write_subarray_pointing(self, event: ArrayEventContainer, writer: TableWriter):
        """ store subarray pointing info in a monitoring table """
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
            container_prefix = ""
            obs_id = Field(0, "Simulation Run Identifier")

        extramc = ExtraSimInfo()
        extramc.obs_id = self._obs_id
        self._simulation_config.prefix = ""
        self._writer.write(
            "configuration/simulation/run", [extramc, self._simulation_config]
        )

    def write_simulation_histograms(self, event_source):
        """Write the distribution of thrown showers

        TODO: this needs to be fixed, since it currently requires access to the
        low-level _file attribute of the SimTelEventSource.  Instead, SimTelEventSource should
        provide this as header info, like `source.simulation_config`

        Notes
        -----
        - this only runs if this is a simulation file. The current implementation is
          a bit of a hack and implies we should improve SimTelEventSource to read this
          info.
        - Currently the histograms are at the end of the simtel file, so if max_events
          is set to non-zero, the end of the file may not be read, and this no
          histograms will be found.

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
            """ fill from a SimTel Histogram entry"""
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
            container.bins_energy = 10 ** ybins * u.TeV
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
                    fill_from_simtel(self._obs_id, hist, hist_container)
                    self._writer.write(
                        table_name="simulation/service/shower_distribution",
                        containers=hist_container,
                    )

    def _write_telescope_events(self, writer: TableWriter, event: ArrayEventContainer):
        """
        add entries to the event/telescope tables for each telescope in a single
        event
        """

        # write the telescope tables

        for tel_id, dl1_camera in event.dl1.tel.items():
            dl1_camera.prefix = ""  # don't want a prefix for this container
            telescope = self._subarray.tel[tel_id]
            tel_type = str(telescope)
            self.log.debug("WRITING TELESCOPE %s: %s", tel_id, telescope)

            tel_index = TelEventIndexContainer(
                obs_id=event.index.obs_id,
                event_id=event.index.event_id,
                tel_id=np.int16(tel_id),
            )

            pnt = event.pointing.tel[tel_id]
            current_pointing = (pnt.azimuth, pnt.altitude)
            if current_pointing != self._last_pointing_tel[tel_id]:
                pnt.prefix = ""
                writer.write(
                    f"dl1/monitoring/telescope/pointing/tel_{tel_id:03d}",
                    [event.trigger.tel[tel_id], pnt],
                )
                self._last_pointing_tel[tel_id] = current_pointing

            table_name = (
                f"tel_{tel_id:03d}" if self.split_datasets_by == "tel_id" else tel_type
            )

            writer.write(
                "dl1/event/telescope/trigger", [tel_index, event.trigger.tel[tel_id]]
            )

            has_sim_camera = (
                tel_id in event.simulation.tel
                and event.simulation.tel[tel_id].true_image.sum() > 0
            )

            if self.write_parameters:
                writer.write(
                    table_name=f"dl1/event/telescope/parameters/{table_name}",
                    containers=[tel_index, *dl1_camera.parameters.values()],
                )

                if self._is_simulation and has_sim_camera:
                    writer.write(
                        f"simulation/event/telescope/parameters/{table_name}",
                        [
                            tel_index,
                            *event.simulation.tel[tel_id].true_parameters.values(),
                        ],
                    )

            if self.write_images:
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

    def _generate_table_indices(self, h5file, start_node):
        """ helper to generate PyTables index tabnles for common columns """
        for node in h5file.iter_nodes(start_node):
            if not isinstance(node, tables.group.Group):
                self.log.debug(f"gen indices for: {node}")
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
        """ generate PyTables index tables for common columns """
        self.log.debug("Writing index tables")
        if self.write_images:
            self._generate_table_indices(
                self._writer._h5file, "/dl1/event/telescope/images"
            )
            if self._is_simulation:
                self._generate_table_indices(
                    self._writer._h5file, "/simulation/event/telescope/images"
                )
        if self.write_parameters:
            self._generate_table_indices(
                self._writer._h5file, "/dl1/event/telescope/parameters"
            )
            if self._is_simulation:
                self._generate_table_indices(
                    self._writer._h5file, "/simulation/event/telescope/parameters"
                )

        self._generate_table_indices(self._writer._h5file, "/dl1/event/subarray")
