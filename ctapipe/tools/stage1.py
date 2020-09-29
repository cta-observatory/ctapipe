"""
Generate DL1 (a or b) output files in HDF5 format from {R0,R1,DL0} inputs.
"""
import pathlib
import sys

import numpy as np
import tables
import tables.filters
from astropy import units as u
from tqdm.autonotebook import tqdm
from collections import defaultdict

from ..io import metadata as meta, DataLevel
from ..calib.camera import CameraCalibrator, GainSelector
from ..containers import (
    ImageParametersContainer,
    TelEventIndexContainer,
    SimulatedShowerDistribution,
    IntensityStatisticsContainer,
    PeakTimeStatisticsContainer,
    MCDL1CameraContainer,
    TimingParametersContainer,
)
from ..core import Provenance
from ..core import QualityQuery, Container, Field, Tool, ToolConfigurationError
from ..core.traits import (
    Bool,
    CaselessStrEnum,
    Int,
    List,
    Path,
    create_class_enum_trait,
    classes_with_traits,
)
from ..image import ImageCleaner
from ..image import (
    hillas_parameters,
    descriptive_statistics,
    concentration as concentration_parameters,
    timing_parameters,
    leakage as leakage_parameters,
    morphology_parameters,
)
from ..image.extractor import ImageExtractor
from ..io import EventSource, HDF5TableWriter, SimTelEventSource

tables.parameters.NODE_CACHE_SLOTS = 3000  # fixes problem with too many datasets

PROV = Provenance()

# define the version of the DL1 data model written here. This should be updated
# when necessary:
# - increase the major number if there is a breaking change to the model
#   (meaning readers need to update scripts)
# - increase the minor number if new columns or datasets are added
# - increase the patch number if there is a small bugfix to the model.
DL1_DATA_MODEL_VERSION = "v1.0.0"


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

    # convert all values to strings, since hdf5 can't handle Times, etc.:
    # TODO: add activity_stop_time?
    headers = {k: str(v) for k, v in reference.to_dict().items()}
    meta.write_to_hdf5(headers, writer._h5file)


class ImageQualityQuery(QualityQuery):
    """ for configuring image-wise data checks """

    quality_criteria = List(
        default_value=[
            ("size_greater_0", "lambda image_selected: image_selected.sum() > 0")
        ],
        help=QualityQuery.quality_criteria.help,
    ).tag(config=True)


class Stage1ProcessorTool(Tool):
    name = "ctapipe-stage1-process"
    description = __doc__ + f" This currently writes {DL1_DATA_MODEL_VERSION} DL1 data"
    examples = """
    To process data with all default values:
    > ctapipe-stage1-process --input events.simtel.gz --output events.dl1.h5 --progress

    Or use an external configuration file, where you can specify all options:
    > ctapipe-stage1-process --config stage1_config.json --progress

    The config file should be in JSON or python format (see traitlets docs). For an
    example, see ctapipe/examples/stage1_config.json in the main code repo.
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

    image_extractor_type = create_class_enum_trait(
        base_class=ImageExtractor,
        default_value="NeighborPeakWindowSum",
        help="Method to use to turn a waveform into a single charge value",
    ).tag(config=True)

    image_cleaner_type = create_class_enum_trait(
        base_class=ImageCleaner, default_value="TailcutsImageCleaner"
    )

    write_index_tables = Bool(
        help=(
            "Generate PyTables index datasets for all tables that contain an "
            "event_id or tel_id. These speed up in-kernal pytables operations,"
            "but add some overhead to the file. They can also be generated "
            "and attached after the file is written "
        ),
        default_value=False,
    ).tag(config=True)

    overwrite = Bool(help="overwrite output file if it exists").tag(config=True)
    progress_bar = Bool(help="show progress bar during processing").tag(config=True)

    aliases = {
        "input": "EventSource.input_url",
        "output": "Stage1ProcessorTool.output_path",
        "allowed-tels": "EventSource.allowed_tels",
        "max-events": "EventSource.max_events",
        "image-extractor-type": "Stage1ProcessorTool.image_extractor_type",
        "image-cleaner-type": "Stage1ProcessorTool.image_cleaner_type",
    }

    flags = {
        "write-images": (
            {"Stage1ProcessorTool": {"write_images": True}},
            "store DL1/Event/Telescope images in output",
        ),
        "write-parameters": (
            {"Stage1ProcessorTool": {"write_parameters": True}},
            "store DL1/Event/Telescope parameters in output",
        ),
        "write-index-tables": (
            {"Stage1ProcessorTool": {"write_index_tables": True}},
            "generate PyTables index tables for the parameter and image datasets",
        ),
        "overwrite": (
            {"Stage1ProcessorTool": {"overwrite": True}},
            "Overwrite output file if it exists",
        ),
        "progress": (
            {"Stage1ProcessorTool": {"progress_bar": True}},
            "show a progress bar during event processing",
        ),
    }

    classes = List(
        [CameraCalibrator, ImageQualityQuery, EventSource]
        + classes_with_traits(ImageCleaner)
        + classes_with_traits(ImageExtractor)
        + classes_with_traits(GainSelector)
    )

    def setup(self):
        # prepare output path:
        self.output_path = self.output_path.expanduser()
        if self.output_path.exists():
            if self.overwrite:
                self.log.warning(f"Overwriting {self.output_path}")
                self.output_path.unlink()
            else:
                self.log.critical(
                    f"Output file {self.output_path} exists"
                    ", use `--overwrite` to overwrite "
                )
                sys.exit(1)

        PROV.add_output_file(str(self.output_path), role="DL1/Event")

        # check that options make sense:
        if self.write_parameters is False and self.write_images is False:
            raise ToolConfigurationError(
                "The options 'write_parameters' and 'write_images' are "
                "both set to False. No output will be generated in that case. "
                "Please enable one or both of these options."
            )

        # setup components:
        self.event_source = self.add_component(EventSource.from_config(parent=self))

        datalevels = self.event_source.datalevels
        if DataLevel.R1 not in datalevels and DataLevel.DL0 not in datalevels:
            self.log.critical(
                f"{self.name} needs the EventSource to provide either R1 or DL0 data"
                f", {self.event_source} provides only {datalevels}"
            )
            sys.exit(1)

        self.image_extractor = self.add_component(
            ImageExtractor.from_name(
                self.image_extractor_type,
                parent=self,
                subarray=self.event_source.subarray,
            )
        )
        self.calibrate = self.add_component(
            CameraCalibrator(
                parent=self,
                subarray=self.event_source.subarray,
                image_extractor=self.image_extractor,
            )
        )
        self.clean = self.add_component(
            ImageCleaner.from_name(
                self.image_cleaner_type,
                parent=self,
                subarray=self.event_source.subarray,
            )
        )
        self.check_image = self.add_component(ImageQualityQuery(parent=self))

        # warn if max_events prevents writing the histograms
        if (
            isinstance(self.event_source, SimTelEventSource)
            and self.event_source.max_events
            and self.event_source.max_events > 0
        ):
            self.log.warning(
                "No Simulated shower distributions will be written because "
                "EventSource.max_events is set to a non-zero number (and therefore "
                "shower distributions read from the input MC file are invalid)."
            )

        # setup HDF5 compression:
        self._hdf5_filters = tables.Filters(
            complevel=self.compression_level,
            complib=self.compression_type,
            fletcher32=True,  # attach a checksum to each chunk for error correction
        )

        # store last pointing to only write unique poitings
        self._last_pointing_tel = defaultdict(lambda: (np.nan * u.deg, np.nan * u.deg))

    def _write_simulation_configuration(self, writer):
        """
        Write the simulation headers to a single row of a table. Later
        if this file is merged with others, that table will grow.

        Note that this function should be run first
        """
        self.log.debug("Writing simulation configuration")

        class ExtraMCInfo(Container):
            container_prefix = ""
            obs_id = Field(0, "MC Run Identifier")

        extramc = ExtraMCInfo()
        extramc.obs_id = self.event_source.obs_id
        self.event_source.mc_header.prefix = ""
        writer.write(
            "configuration/simulation/run", [extramc, self.event_source.mc_header]
        )

    def _write_simulation_histograms(self, writer: HDF5TableWriter):
        """ Write the distribution of thrown showers

        Notes
        -----
        - this only runs if this is a simulation file. The current implementation is
          a bit of a hack and implies we should improve SimTelEventSource to read this
          info.
        - Currently the histograms are at the end of the simtel file, so if max_events
          is set to non-zero, the end of the file may not be read, and this no
          histograms will be found.
        """
        self.log.debug("Writing simulation histograms")

        if not isinstance(self.event_source, SimTelEventSource):
            return

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

        hists = self.event_source.file_.histograms
        if hists is not None:
            hist_container = SimulatedShowerDistribution()
            hist_container.prefix = ""
            for hist in hists:
                if hist["id"] == 6:
                    fill_from_simtel(self.event_source.obs_id, hist, hist_container)
                    writer.write(
                        table_name="simulation/service/shower_distribution",
                        containers=hist_container,
                    )

    def _write_processing_statistics(self):
        """ write out the event selection stats, etc. """
        image_stats = self.check_image.to_table(functions=True)
        image_stats.write(
            self.output_path,
            path="/dl1/service/image_statistics",
            append=True,
            serialize_meta=True,
        )

    def _parameterize_image(self, tel_id, image, signal_pixels, peak_time=None):
        """Apply image cleaning and calculate image features

        Parameters
        ----------
        subarray : SubarrayDescription
           subarray description
        data : DL1CameraContainer
            calibrated camera data
        tel_id: int
            which telescope is being cleaned

        Returns
        -------
        np.ndarray, ImageParametersContainer:
            cleaning mask, parameters
        """

        tel = self.event_source.subarray.tel[tel_id]
        geometry = tel.camera.geometry
        image_selected = image[signal_pixels]

        # check if image can be parameterized:
        image_criteria = self.check_image(image_selected)
        self.log.debug(
            "image_criteria: %s",
            list(zip(self.check_image.criteria_names[1:], image_criteria)),
        )

        # parameterize the event if all criteria pass:
        if all(image_criteria):
            geom_selected = geometry[signal_pixels]

            hillas = hillas_parameters(geom=geom_selected, image=image_selected)
            leakage = leakage_parameters(
                geom=geometry, image=image, cleaning_mask=signal_pixels
            )
            concentration = concentration_parameters(
                geom=geom_selected, image=image_selected, hillas_parameters=hillas
            )
            morphology = morphology_parameters(geom=geometry, image_mask=signal_pixels)
            intensity_statistics = descriptive_statistics(
                image_selected, container_class=IntensityStatisticsContainer
            )

            if peak_time is not None:
                timing = timing_parameters(
                    geom=geom_selected,
                    image=image_selected,
                    peak_time=peak_time[signal_pixels],
                    hillas_parameters=hillas,
                )
                peak_time_statistics = descriptive_statistics(
                    peak_time[signal_pixels],
                    container_class=PeakTimeStatisticsContainer,
                )
            else:
                timing = TimingParametersContainer()
                peak_time_statistics = PeakTimeStatisticsContainer()

            return ImageParametersContainer(
                hillas=hillas,
                timing=timing,
                leakage=leakage,
                morphology=morphology,
                concentration=concentration,
                intensity_statistics=intensity_statistics,
                peak_time_statistics=peak_time_statistics,
            )

        # return the default container (containing nan values) for no
        # parameterization
        return ImageParametersContainer()

    def _process_events(self, writer):
        self.log.debug("Writing DL1/Event data")
        self.event_source.subarray.info(printer=self.log.debug)

        # initial value for last known pointing position
        last_pointing = (np.nan * u.deg, np.nan * u.deg)

        for event in tqdm(
            self.event_source,
            desc=self.event_source.__class__.__name__,
            total=self.event_source.max_events,
            unit="ev",
            disable=not self.progress_bar,
        ):

            self.log.log(9, "Writing event_id=%s", event.index.event_id)

            self.calibrate(event)

            event.trigger.prefix = ""

            p = event.pointing
            current_pointing = (p.array_azimuth, p.array_altitude)
            if current_pointing != last_pointing:
                p.prefix = ""
                writer.write("dl1/monitoring/subarray/pointing", [event.trigger, p])
                last_pointing = current_pointing

            # write the subarray tables
            if self.event_source.is_simulation:
                writer.write(
                    table_name="simulation/event/subarray/shower",
                    containers=[event.index, event.mc],
                )
            writer.write(
                table_name="dl1/event/subarray/trigger",
                containers=[event.index, event.trigger],
            )
            # write the telescope tables
            self._write_telescope_event(writer, event)

    def _write_telescope_event(self, writer, event):
        """
        add entries to the event/telescope tables for each telescope in a single
        event
        """
        # write the telescope tables
        for tel_id, dl1_camera in event.dl1.tel.items():
            dl1_camera.prefix = ""  # don't want a prefix for this container
            telescope = self.event_source.subarray.tel[tel_id]
            tel_type = str(telescope)

            tel_index = TelEventIndexContainer(
                obs_id=event.index.obs_id,
                event_id=event.index.event_id,
                tel_id=np.int16(tel_id),
            )

            p = event.pointing.tel[tel_id]
            current_pointing = (p.azimuth, p.altitude)
            if current_pointing != self._last_pointing_tel[tel_id]:
                p.prefix = ""
                writer.write(
                    f"dl1/monitoring/telescope/pointing/tel_{tel_id:03d}",
                    [event.trigger.tel[tel_id], p],
                )
                self._last_pointing_tel[tel_id] = current_pointing

            table_name = (
                f"tel_{tel_id:03d}" if self.split_datasets_by == "tel_id" else tel_type
            )

            writer.write(
                "dl1/event/telescope/trigger", [tel_index, event.trigger.tel[tel_id]]
            )

            if self.event_source.is_simulation:
                true_image = event.mc.tel[tel_id].true_image
                has_true_image = (
                    true_image is not None and np.count_nonzero(true_image) > 0
                )

                if has_true_image:
                    mcdl1 = MCDL1CameraContainer(
                        true_image=true_image, true_parameters=None
                    )
                    mcdl1.prefix = ""
            else:
                has_true_image = False

            if self.write_parameters:
                # apply cleaning
                dl1_camera.image_mask = self.clean(
                    tel_id=tel_id,
                    image=dl1_camera.image,
                    arrival_times=dl1_camera.peak_time,
                )

                params = self._parameterize_image(
                    tel_id=tel_id,
                    image=dl1_camera.image,
                    signal_pixels=dl1_camera.image_mask,
                    peak_time=dl1_camera.peak_time,
                )

                self.log.debug("params: %s", params.as_dict(recursive=True))
                writer.write(
                    table_name=f"dl1/event/telescope/parameters/{table_name}",
                    containers=[tel_index, *params.values()],
                )

                if self.event_source.is_simulation and has_true_image:
                    mcdl1.true_parameters = self._parameterize_image(
                        tel_id,
                        image=true_image,
                        signal_pixels=true_image > 0,
                        peak_time=None,  # true image from mc has no peak time
                    )
                    writer.write(
                        f"simulation/event/telescope/parameters/{table_name}",
                        [tel_index, *mcdl1.true_parameters.values()],
                    )

            if self.write_images:
                # note that we always write the image, even if the image quality
                # criteria are not met (those are only to determine if the parameters
                # can be computed).
                writer.write(
                    table_name=f"dl1/event/telescope/images/{table_name}",
                    containers=[tel_index, dl1_camera],
                )

                if has_true_image:
                    writer.write(
                        f"simulation/event/telescope/images/{table_name}",
                        [tel_index, mcdl1],
                    )

    def _generate_table_indices(self, h5file, start_node):

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

    def _generate_indices(self, writer):

        if self.write_images:
            self._generate_table_indices(writer._h5file, "/dl1/event/telescope/images")
        self._generate_table_indices(writer._h5file, "/dl1/event/subarray")

    def _setup_writer(self, writer: HDF5TableWriter):
        writer.add_column_transform(
            table_name="dl1/event/subarray/trigger",
            col_name="tels_with_trigger",
            transform=self.event_source.subarray.tel_ids_to_mask,
        )

        # exclude some columns that are not writable
        writer.exclude("dl1/event/subarray/trigger", "tel")
        writer.exclude("dl1/monitoring/subarray/pointing", "tel")
        writer.exclude("dl1/monitoring/subarray/pointing", "event_type")
        for tel_id, telescope in self.event_source.subarray.tel.items():
            tel_type = str(telescope)
            if self.split_datasets_by == "tel_id":
                table_name = f"tel_{tel_id:03d}"
            else:
                table_name = tel_type

            if self.write_parameters is False:
                writer.exclude(
                    f"/dl1/event/telescope/images/{table_name}", "image_mask"
                )
            writer.exclude(f"/dl1/event/telescope/images/{table_name}", "parameters")
            writer.exclude(
                f"/dl1/monitoring/event/pointing/tel_{tel_id:03d}", "event_type"
            )
            if self.event_source.is_simulation:
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

    def start(self):

        # FIXME: this uses astropy tables hdf5 io, internally using h5py,
        # and must thus be done before the table writer opens the file or it might lead
        # to "Resource temporary unavailable" if h5py and tables are not linked
        # against the same libhdf (happens when using the pre-build pip wheels)
        # should be replaced by writing the table using tables
        self.event_source.subarray.to_hdf(self.output_path)

        with HDF5TableWriter(
            self.output_path,
            parent=self,
            mode="a",
            add_prefix=True,
            filters=self._hdf5_filters,
        ) as writer:

            if self.event_source.is_simulation:
                self._write_simulation_configuration(writer)

            self._setup_writer(writer)
            self._process_events(writer)

            if self.event_source.is_simulation:
                self._write_simulation_histograms(writer)

            if self.write_index_tables:
                self._generate_indices(writer)

            write_reference_metadata_headers(
                subarray=self.event_source.subarray,
                obs_id=self.event_source.obs_id,
                writer=writer,
            )
        self._write_processing_statistics()

    def finish(self):
        pass


def main():
    tool = Stage1ProcessorTool()
    tool.run()


if __name__ == "__main__":
    main()
