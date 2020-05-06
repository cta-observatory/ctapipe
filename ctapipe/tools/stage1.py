"""
Generate DL1 (a or b) output files in HDF5 format from {R0,R1,DL0} inputs.

# TODO: add event time per telescope!
"""
import hashlib
from functools import partial
from os.path import expandvars
from pathlib import Path

import numpy as np
import tables
import tables.filters

from astropy import units as u
from tqdm.autonotebook import tqdm

from ctapipe.io import metadata as meta
from ..calib.camera import CameraCalibrator, GainSelector
from ..containers import (
    ImageParametersContainer,
    TelEventIndexContainer,
    SimulatedShowerDistribution,
    MorphologyContainer,
    IntensityStatisticsContainer,
    PeakTimeStatisticsContainer,
)
from ..core import Provenance
from ..core import QualityQuery, Container, Field, Tool, ToolConfigurationError
from ..core.traits import (
    Bool,
    CaselessStrEnum,
    Int,
    List,
    Unicode,
    create_class_enum_trait,
    classes_with_traits,
)
from ..image import ImageCleaner
from ..image import (
    hillas_parameters,
    number_of_islands,
    number_of_island_sizes,
    descriptive_statistics,
)
from ..image.concentration import concentration
from ..image.extractor import ImageExtractor
from ..image.leakage import leakage
from ..image.timing_parameters import timing_parameters
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


def write_reference_metadata_headers(output_filename, obs_id, subarray, writer):
    """
    Attaches Core Provenence headers to an output HDF5 file.
    Right now this is hard-coded for use with the ctapipe-stage1-process tool

    Parameters
    ----------
    output_filename: str
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
        contact=meta.Contact(
            name="", email="", organization="CTA Consortium"                                               "Consortium"
        ),
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
            site="Other", # need a way to detect site...
            class_="Subarray",
            type_="unknown",
            version="unknown",
            id_=subarray.name,
        ),
    )

    # convert all values to strings, since hdf5 can't handle Times, etc.:
    # TODO: add activity_stop_time?
    headers = {k:str(v) for k,v in reference.to_dict().items()}
    meta.write_to_hdf5(headers, writer._h5file)


def morphology(geom, image_mask) -> MorphologyContainer:
    """
    Compute image morphology parameters
    Parameters
    ----------
    geom: ctapipe.instrument.camera.CameraGeometry
        camera description
    image_mask: np.ndarray(bool)
        image of pixels surviving cleaning (True=survives)
    Returns
    -------
    MorphologyContainer:
        parameters related to the morphology
    """

    num_islands, island_labels = number_of_islands(geom=geom, mask=image_mask)

    n_small, n_medium, n_large = number_of_island_sizes(island_labels)

    return MorphologyContainer(
        num_pixels=np.count_nonzero(image_mask),
        num_islands=num_islands,
        num_small_islands=n_small,
        num_medium_islands=n_medium,
        num_large_islands=n_large,
    )


class ExtendedImageParametersContainer(ImageParametersContainer):
    """
    Extra parameters to add to the ImageParametersContainer

    TODO: should eventually just move to ImageParametersContainer.
    """
    mc_intensity = Field(IntensityStatisticsContainer(), "MC intensity statistics")


class ExtraImageContainer(Container):
    """
    Extra information to attach to the image dataset

    TODO: update MCCameraEventContainer and DL1TelescopeContainer; remove this
    """

    container_prefix = ""

    true_image = Field(
        None, "Monte-carlo image of photo electrons on the camera plane, without noise"
    )

    image_mask = Field(None, "Boolean array of pixels, True=used in parameterization")
    selected_gain_channel = Field(None, "Array [n_pix] of gain channel used")


def tel_type_string_to_int(tel_type):
    """
    convert a telescope type string (str(TelescopeDescription)) into an integer that
    can be stored.

    Parameters
    ----------
    tel_type: str
        telescope type string like "SST_ASTRI_CHEC"

    Returns
    -------
    int:
        hash value
    """
    return np.int32(
        int(hashlib.sha1(tel_type.encode("utf8")).hexdigest(), 16) % (10 ** 8)
    )


class ImageQualityQuery(QualityQuery):
    """ for configuring image-wise data checks """
    pass


def expand_tel_list(tel_list, max_tels, index_map):
    """
    un-pack var-length list of tel_ids into
    fixed-width bit pattern by tel_index

    TODO: use index_map to index by tel_index rather than tel_id so this can be a
    shorter array of bools.
    """
    pattern = np.zeros(max_tels).astype(bool)
    pattern[tel_list] = 1
    return pattern


def create_tel_id_to_tel_index_transform(sub):
    """
    build a mapping of tel_id back to tel_index:
    (note this should be part of SubarrayDescription)
    """
    idx = np.zeros(max(sub.tel_indices) + 1)
    for key, val in sub.tel_indices.items():
        idx[key] = val

    # the final transform then needs the mapping and the number of telescopes
    return partial(expand_tel_list, max_tels=len(sub.tel) + 1, index_map=idx)


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

    output_filename = Unicode(
        help="DL1 output filename", default_value="events.dl1.h5"
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
        default_value="zlib",
    ).tag(config=True)

    image_extractor_type = create_class_enum_trait(
        base_class=ImageExtractor,
        default_value="NeighborPeakWindowSum",
        help="Method to use to turn a waveform into a single charge value",
    ).tag(config=True)

    gain_selector_type = create_class_enum_trait(
        base_class=GainSelector, default_value="ThresholdGainSelector"
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
        "output": "Stage1ProcessorTool.output_filename",
        "allowed-tels": "EventSource.allowed_tels",
        "max-events": "EventSource.max_events",
        "image-extractor-type": "Stage1ProcessorTool.image_extractor_type",
        "gain-selector-type": "Stage1ProcessorTool.gain_selector_type",
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
        [EventSource, CameraCalibrator, ImageQualityQuery]
        + classes_with_traits(ImageCleaner)
        + classes_with_traits(ImageExtractor)
        + classes_with_traits(GainSelector)
    )

    def setup(self):

        # prepare output path:

        output_path = Path(expandvars(self.output_filename)).expanduser()
        if output_path.exists() and self.overwrite:
            self.log.warning(f"Overwriting {output_path}")
            output_path.unlink()
        PROV.add_output_file(str(output_path), role="DL1/Event")

        # check that options make sense:
        if self.write_parameters is False and self.write_images is False:
            raise ToolConfigurationError(
                "The options 'write_parameters' and 'write_images' are "
                "both set to False. No output will be generated in that case. "
                "Please enable one or both of these options."
            )

        # setup components:

        self.gain_selector = self.add_component(
            GainSelector.from_name(self.gain_selector_type, parent=self)
        )
        self.event_source = self.add_component(
            EventSource.from_config(parent=self, gain_selector=self.gain_selector)
        )
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

        # check component setup
        if self.event_source.max_events and self.event_source.max_events > 0:
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

    def _write_simulation_configuration(self, writer, event):
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
        extramc.obs_id = event.index.obs_id
        event.mcheader.prefix = ""
        writer.write("configuration/simulation/run_config", [extramc, event.mcheader])

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

        if type(self.event_source) is not SimTelEventSource:
            return

        hists = self.event_source.file_.histograms
        if hists is not None:
            hist_container = SimulatedShowerDistribution()
            hist_container.prefix = ""
            for hist in hists:
                if hist["id"] == 6:
                    fill_from_simtel(self._cur_obs_id, hist, hist_container)
                    writer.write(
                        table_name="configuration/simulation/shower_distribution",
                        containers=hist_container,
                    )

    def _write_instrument_configuration(self, subarray):
        """write the SubarrayDescription

        Parameters
        ----------
        subarray : ctapipe.instrument.SubarrayDescription
            subarray description
        """
        self.log.debug("Writing instrument configuration")
        serialize_meta = True

        subarray.to_table().write(
            self.output_filename,
            path="/configuration/instrument/subarray/layout",
            serialize_meta=serialize_meta,
            append=True,
        )
        subarray.to_table(kind="optics").write(
            self.output_filename,
            path="/configuration/instrument/telescope/optics",
            append=True,
            serialize_meta=serialize_meta,
        )
        for telescope_type in subarray.telescope_types:
            ids = set(subarray.get_tel_ids_for_type(telescope_type))
            if len(ids) > 0:  # only write if there is a telescope with this camera
                tel_id = list(ids)[0]
                camera = subarray.tel[tel_id].camera
                camera.geometry.to_table().write(
                    self.output_filename,
                    path=f"/configuration/instrument/telescope/camera/geometry_{camera}",
                    append=True,
                    serialize_meta=serialize_meta,
                )
                camera.readout.to_table().write(
                    self.output_filename,
                    path=f"/configuration/instrument/telescope/camera/readout_{camera}",
                    append=True,
                    serialize_meta=serialize_meta,
                )

    def _write_processing_statistics(self):
        """ write out the event selection stats, etc. """
        image_stats = self.check_image.to_table(functions=True)
        image_stats.write(
            self.output_filename,
            path="/dl1/service/image_statistics",
            append=True,
            serialize_meta=True,
        )

    def _parameterize_image(self, subarray, data, tel_id):
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

        tel = subarray.tel[tel_id]
        geometry = tel.camera.geometry

        # apply cleaning
        signal_pixels = self.clean(
            tel_id=tel_id, image=data.image, arrival_times=data.peak_time
        )
        image_selected = data.image[signal_pixels]

        params = ExtendedImageParametersContainer()

        # check if image can be parameterized:
        image_criteria = self.check_image(image_selected)
        self.log.debug(
            "image_criteria: %s",
            list(zip(self.check_image.criteria_names[1:], image_criteria)),
        )

        # parameterize the event if all criteria pass:
        if all(image_criteria):
            geom_selected = geometry[signal_pixels]

            params.hillas = hillas_parameters(
                geom=geom_selected, image=image_selected,
            )
            params.timing = timing_parameters(
                geom=geom_selected,
                image=image_selected,
                peak_time=data.peak_time[signal_pixels],
                hillas_parameters=params.hillas,
            )
            params.leakage = leakage(
                geom=geometry, image=data.image, cleaning_mask=signal_pixels
            )
            params.concentration = concentration(
                geom=geom_selected,
                image=image_selected,
                hillas_parameters=params.hillas,
            )
            params.morphology = morphology(
                geom=geometry, image_mask=signal_pixels
            )
            params.intensity_statistics = descriptive_statistics(
                image_selected, container_class=IntensityStatisticsContainer
            )
            params.peak_time_statistics = descriptive_statistics(
                data.peak_time[signal_pixels],
                container_class=PeakTimeStatisticsContainer,
            )

        return signal_pixels, params

    def _process_events(self, writer):
        self.log.debug("Writing DL1/Event data")
        is_initialized = False
        self.event_source.subarray.info(printer=self.log.debug)

        for event in tqdm(
            self.event_source,
            desc=self.event_source.__class__.__name__,
            total=self.event_source.max_events,
            unit="ev",
            disable=not self.progress_bar,
        ):

            if not is_initialized:
                self._write_simulation_configuration(writer, event)
                is_initialized = True

            self.log.log(9, "Writing event_id=%s", event.index.event_id)

            self.calibrate(event)

            event.mc.prefix = "mc"
            event.trig.prefix = ""
            self._cur_obs_id = event.index.obs_id

            # write the subarray tables
            writer.write(
                table_name="dl1/event/subarray/mc_shower",
                containers=[event.index, event.mc],
            )
            writer.write(
                table_name="dl1/event/subarray/trigger",
                containers=[event.index, event.trig],
            )
            # write the telescope tables
            self._write_telescope_event(writer, event)

        if is_initialized is False:
            raise ValueError(f"No events found in file: {self.event_source.input_url}")

    def _write_telescope_event(self, writer, event):
        """
        add entries to the event/telescope tables for each telescope in a single
        event
        """

        # write the telescope tables
        for tel_id, data in event.dl1.tel.items():

            data.prefix = ""  # don't want a prefix for this container
            telescope = self.event_source.subarray.tel[tel_id]
            tel_type = str(telescope)

            tel_index = TelEventIndexContainer(
                **event.index,
                tel_id=np.int16(tel_id),
                tel_type_id=tel_type_string_to_int(tel_type)
            )
            table_name = (
                f"tel_{tel_id:03d}" if self.split_datasets_by == "tel_id" else tel_type
            )

            extra = ExtraImageContainer(
                true_image=event.mc.tel[tel_id].true_image,
                selected_gain_channel=event.r1.tel[tel_id].selected_gain_channel,
                image_mask=None,  # added later, if computed only
            )

            if self.write_parameters:

                image_mask, params = self._parameterize_image(
                    self.event_source.subarray, data, tel_id=tel_id
                )

                self.log.debug("params: %s", params.as_dict(recursive=True))

                containers_to_write = [
                    tel_index,
                    params.hillas,
                    params.timing,
                    params.leakage,
                    params.concentration,
                    params.morphology,
                    params.intensity_statistics,
                    params.peak_time_statistics,
                ]

                # currently the HDF5TableWriter has problems if the first event
                # has NaN as values, since it can't infer the data types.
                # that implies we need to specify them in the Fields, rather than
                # infer from first event, perhaps.  For now we skip them.
                parameters_were_computed = (
                    False if params.hillas.intensity is np.nan else True
                )

                if parameters_were_computed:
                    writer.write(
                        table_name=f"dl1/event/telescope/parameters/{table_name}",
                        containers=containers_to_write,
                    )
                extra.image_mask = image_mask

            if self.write_images:
                # note that we always write the image, even if the image quality
                # criteria are not met (those are only to determine if the parameters
                # can be computed).
                writer.write(
                    table_name=f"dl1/event/telescope/images/{table_name}",
                    containers=[tel_index, data, extra],
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

    def start(self):

        # FIXME: this uses astropy tables hdf5 io, internally using h5py,
        # and must thus be done before the table writer opens the file or it might lead
        # to "Resource temporary unavailable" if h5py and tables are not linked
        # against the same libhdf (happens when using the pre-build pip wheels)
        # should be replaced by writing the table using tables
        self._write_instrument_configuration(self.event_source.subarray)

        with HDF5TableWriter(
            self.output_filename, mode="a", add_prefix=True, filters=self._hdf5_filters
        ) as writer:

            tel_list_transform = create_tel_id_to_tel_index_transform(
                self.event_source.subarray
            )
            writer.add_column_transform(
                table_name="dl1/event/subarray/trigger",
                col_name="tels_with_trigger",
                transform=tel_list_transform,
            )
            if self.write_parameters is False:
                # don't need to write out the image mask if no parameters are computed,
                # since we don't do image cleaning in that case.
                writer.exclude("/dl1/event/telescope/images", "image_mask")

            self._process_events(writer)
            self._write_simulation_histograms(writer)
            # self._write_processing_statistics()

            if self.write_index_tables:
                self._generate_indices(writer)

            write_reference_metadata_headers(
                output_filename=self.output_filename,
                subarray=self.event_source.subarray,
                obs_id=self._cur_obs_id,
                writer=writer,
            )

    def finish(self):
        pass


def main():
    tool = Stage1ProcessorTool()
    tool.run()


if __name__ == "__main__":
    main()
