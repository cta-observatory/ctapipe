""" 
Generate DL1 (a or b) output files in HDF5 format from {R0,R1,DL0} inputs.
"""
import hashlib
from collections import namedtuple
from functools import partial
from os.path import expandvars
from pathlib import Path

import numpy as np
import tables.filters
from astropy import units as u
from ctapipe.calib.camera import CameraCalibrator, GainSelector
from ctapipe.core import Component, Container, Field, Tool, ToolConfigurationError
from ctapipe.core import Provenance
from ctapipe.core.traits import (
    Bool,
    CaselessStrEnum,
    Dict,
    Int,
    List,
    Unicode,
    enum_trait,
    classes_with_traits,
    IntTelescopeParameter,
    FloatTelescopeParameter,
)
from ctapipe.image import hillas_parameters, tailcuts_clean, number_of_islands
from ctapipe.image.concentration import concentration
from ctapipe.image.extractor import ImageExtractor
from ctapipe.image.leakage import leakage
from ctapipe.image.timing_parameters import timing_parameters
from ctapipe.io import EventSource, HDF5TableWriter, SimTelEventSource
from ctapipe.io.containers import (
    DL1CameraContainer,
    EventIndexContainer,
    ImageParametersContainer,
    TelEventIndexContainer,
    SimulatedShowerDistribution,
    MorphologyContainer,
)
from tqdm.autonotebook import tqdm

PROV = Provenance()

# define the version of the DL1 data model written here. This should be updated
# when necessary:
# - increase the major number if there is a breaking change to the model
#   (meaning readers need to update scripts)
# - increase the minor number if new columns or datasets are added
# - increase the patch number if there is a small bugfix to the model.
DL1_DATA_MODEL_VERSION = "v2.0.0"
EXAMPLE_CONFIG = """
{
    "Stage1Process": {
        "config_file": "",
        "output_filename": "lapalma_proton_small.h5",
        "overwrite": true,
        "write_images": true,
        "image_extractor_type": "NeighborPeakWindowSum",
        "image_cleaner_type": "TailcutsImageCleaner"
    },
    "EventSource": {
        "allowed_tels": [],
        "input_url": "~/Data/CTA/Prod3/LaPalmaRefSim/proton_20deg_180deg_run18___cta-prod3-demo-2147m-LaPalma-baseline.simtel.gz",
        "skip_calibration_events": true
    },
    "TailcutsImageCleaner": {
        "boundary_threshold_pe": [
            ["type","*", 5.0],
            ["type", "LST*", 3.0],
            ["type", "MST*", 4.0]
        ],
        "min_picture_neighbors":[
            ["type","*",2]
        ]
    ,
        "picture_threshold_pe":  [
                ["type", "*", 10.0],
                ["type", "LST_LST_LSTCam", 6.0],
                ["type", "MST_MST_NectarCam", 6.0],
                ["id", 12, 15.0]
        ]
    },
    "ImageDataChecker": {
        "selection_functions": {
            "enough_pixels": "lambda im: np.count_nonzero(im) > 2",
            "enough_charge": "lambda im: im.sum() > 100"
        }
    }
}
"""

def write_core_provenance(output_filename, obs_id, subarray):
    """
    Attaches Core Provenence headers to an output HDF5 file.
    Right now this is hard-coded for use with the ctapipe-stage1-process tool

    TODO: generalize, and move to ctapipe.io

    Parameters
    ----------
    output_filename: str
        output HDF5 file
    obs_id: int
        observation ID
    subarray:
        SubarrayDescription to get metadata from
    """
    import uuid

    activity = PROV.current_activity.provenance

    metadata = {
        "CTA METADATA VERSION ": "2",
        "CTA CONTACT ORGANIZATION": "CTA Consortium",
        "CTA CONTACT NAME": "K. Kosack",
        "CTA CONTACT EMAIL": "karl.kosack@cea.fr",
        "CTA PRODUCT DESCRIPTION": "DL1 Event List",
        "CTA PRODUCT CREATION_TIME": "2018-11-10 15:30:00",
        "CTA PRODUCT ID": str(uuid.uuid4()),
        "CTA PRODUCT DATA CATEGORY": "MC",
        "CTA PRODUCT DATA LEVEL": "DL1",
        "CTA PRODUCT DATA TYPE": "Event",
        "CTA PRODUCT DATA ASSOCIATION": "Subarray",
        "CTA PRODUCT DATA MODEL NAME": "DL1/Event",
        "CTA PRODUCT DATA MODEL VERSION": DL1_DATA_MODEL_VERSION,
        "CTA PRODUCT DATA MODEL URL": "",
        "CTA PRODUCT FORMAT": "hdf5",
        "CTA PROCESS TYPE": "simulation",
        "CTA PROCESS SUBTYPE": "",
        "CTA PROCESS ID": str(obs_id),
        "CTA ACTIVITY NAME": activity["activity_name"],
        "CTA ACTIVITY TYPE": "software",
        "CTA ACTIVITY ID": activity["activity_uuid"],
        "CTA ACTIVITY START": activity["start"]["time_utc"],
        "CTA ACTIVITY SOFTWARE NAME": "ctapipe",
        "CTA ACTIVITY SOFTWARE VERSION": activity["system"]["ctapipe_version"],
        "CTA INSTRUMENT SITE": "unknown",
        "CTA INSTRUMENT CLASS": "subarray",
        "CTA INSTRUMENT TYPE": "",
        "CTA INSTRUMENT SUBTYPE": "",
        "CTA INSTRUMENT VERSION": "",
        "CTA INSTRUMENT ID": str(subarray),
    }

    with tables.open_file(output_filename, mode="a") as h5file:
        for key, value in metadata.items():
            h5file.root._v_attrs[key] = value


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

    return MorphologyContainer(num_pixels=image_mask.sum(), num_islands=num_islands)


class IntensityContainer(Container):
    """ Store statistics on the intensity distribution of images """

    max = Field(np.nan, "value of pixel with maximum intensity")
    min = Field(np.nan, "value of pixel with minimum intensity")
    mean = Field(np.nan, "mean intensity")
    std = Field(np.nan, "standard deviation of intensity")


def intensity_statistics(image) -> IntensityContainer:
    """ compute intensity statistics of an image  """
    return IntensityContainer(
        max=image.max(), min=image[image > 0].min(), mean=image.mean(), std=image.std()
    )


class ExtendedImageParametersContainer(ImageParametersContainer):
    """
    Extra parameters to add to the ImageParametersContainer

    TODO: should eventually just move to ImageParametersContainer.
    """

    intensity = Field(IntensityContainer(), "intensity statistics")
    mc_intensity = Field(IntensityContainer(), "MC intensity statistics")


class ExtraImageContainer(Container):
    """
    Extra information to attach to the image dataset

    TODO: update MCCameraEventContainer and DL1TelescopeContainer; remove this
    """

    container_prefix = ""

    mc_photo_electron_image = Field(
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


class DataChecker(Component):
    """
    Manages a set of selection criteria that operate on the same type of input.
    Each time it is called, it returns a boolean array of whether or not each
    criterion passed.
    """

    selection_functions = Dict(
        help=(
            "dict of '<cut name>' : lambda function in string format to accept ("
            "select) a given data value.  E.g. `{'mycut': 'lambda x: x > 3'}` "
        )
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)

        # add a selection to count all entries and make it the first one
        selection_functions = {"TOTAL": "lambda x: True"}
        selection_functions.update(self.selection_functions)

        self.selection_functions = selection_functions  # update

        # generate real functions from the selection function strings
        self._selectors = {
            name: eval(func_str) for name, func_str in selection_functions.items()
        }

        # arrays for recording overall statistics
        self._counts = np.zeros(len(self._selectors), dtype=np.int)
        self._cum_counts = np.zeros(len(self._selectors), dtype=np.int)

        # generate a Container that we can use for output somehow...
        self._container = Container()

    def __len__(self):
        return self._counts[0]

    @property
    def criteria_names(self):
        return list(self._selectors.keys())

    @property
    def selection_function_strings(self):
        return list(self.selection_functions.values())

    def to_table(self, functions=False):
        """
        Return a tabular view of the latest quality summary

        The columns are
        - *criteria*: name of each criterion
        - *counts*: counts of each criterion independently
        - *cum_counts*: counts of cumulative application of each criterion in order

        Parameters
        ----------
        functions: bool:
            include the function string as a column

        Returns
        -------
        astropy.table.Table
        """
        from astropy.table import Table

        cols = {
            "criteria": self.criteria_names,
            "counts": self._counts,
            "cum_counts": self._cum_counts,
        }
        if functions:
            cols["func"] = self.selection_function_strings
        return Table(cols)

    def _repr_html_(self):
        return self.to_table()._repr_html_()

    def __call__(self, value):
        """
        Test that value passes all cuts

        Parameters
        ----------
        value:
            the value to pass to each selection function

        Returns
        -------
        np.ndarray:
            array of booleans with results of each selection criterion in order
        """
        result = np.array(list(map(lambda f: f(value), self._selectors.values())))
        self._counts += result.astype(int)
        self._cum_counts += result.cumprod()
        return result


class ImageDataChecker(DataChecker):
    """ for configuring image-wise data checks """
    pass


class EventDataChecker(DataChecker):
    """ for configuring event-wise data checks """

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


class ImageCleaner(Component):
    pass


class TailcutsImageCleaner(ImageCleaner):
    """
    Apply Image Cleaning to a set of telescope images
    """

    picture_threshold_pe = FloatTelescopeParameter(
        help="top-level threshold in photoelectrons",
        default_value=[("type", "*", 10.0)],
    ).tag(config=True)

    boundary_threshold_pe = FloatTelescopeParameter(
        help="second-level threshold in photoelectrons",
        default_value=[("type", "*", 5.0)],
    ).tag(config=True)

    min_picture_neighbors = IntTelescopeParameter(
        help="Minimum number of neighbors above threshold to consider",
        default_value=[("type", "*", 2)],
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config, parent, **kwargs)
        self._subarray_initialized = False

    def __call__(
        self,
        tel_id: int,
        subarray: "ctapipe.instrument.SubarrayDescription",
        image: np.ndarray,
    ) -> np.ndarray:
        """ Apply image cleaning
        
        Parameters
        ----------
        tel_id: int
            which telescope id in the subarray is being used (determines
            which cut is used)
        image : np.ndarray
            image pixel data corresponding to the camera geometry
        subarray: ctapipe.image.SubarrayDescription
            subarray definition (for mapping tel type to tel_id)
        
        Returns
        -------
        np.ndarray
            boolean mask of pixels passing cleaning
        """
        if not self._subarray_initialized:
            self.picture_threshold_pe.attach_subarray(subarray)
            self.boundary_threshold_pe.attach_subarray(subarray)
            self.min_picture_neighbors.attach_subarray(subarray)
            self._subarray_initialized = True

        return tailcuts_clean(
            subarray.tel[tel_id].camera,
            image,
            picture_thresh=self.picture_threshold_pe[tel_id],
            boundary_thresh=self.boundary_threshold_pe[tel_id],
            min_number_picture_neighbors=self.min_picture_neighbors[tel_id],
        )


class Stage1Process(Tool):
    name = "ctapipe-stage1-process"
    description = __doc__ + f" This currently writes {DL1_DATA_MODEL_VERSION} DL1 data"
    examples = """
    > ctapipe-stage1-process --input events.simtel.gz --output events.dl1.h5 --progress
    
    Or use an external configuration file:
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

    compression_type = CaselessStrEnum(
        values=["blosc:zstd", "zlib"],
        help="compressor algorithm to use. ",
        default_value="zlib",
    ).tag(config=True)

    image_extractor_type = enum_trait(
        base_class=ImageExtractor,
        default="NeighborPeakWindowSum",
        help_str="Method to use to turn a waveform into a single charge value",
    ).tag(config=True)

    gain_selector_type = enum_trait(
        base_class=GainSelector, default="ThresholdGainSelector"
    ).tag(config=True)

    image_cleaner_type = enum_trait(
        base_class=ImageCleaner, default="TailcutsImageCleaner"
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
        "output": "Stage1Process.output_filename",
        "allowed-tels": "EventSource.allowed_tels",
        "max-events": "EventSource.max_events",
        "image-extractor-type": "Stage1Process.image_extractor_type",
        "gain-selector-type": "Stage1Process.gain_selector_type",
        "image-cleaner-type": "Stage1Process.image_cleaner_type",
    }

    flags = {
        "write-images": (
            {"Stage1Process": {"write_images": True}},
            "store DL1/Event/Telescope images in output",
        ),
        "write-parameters": (
            {"Stage1Process": {"write_images": True}},
            "store DL1/Event/Telescope parameters in output",
        ),
        "write-index-tables": (
            {"Stage1Process": {"write_index_tables": True}},
            "generate PyTables index tables for the parameter and image datasets",
        ),
        "overwrite": (
            {"Stage1Process": {"overwrite": True}},
            "Overwrite output file if it exists",
        ),
        "progress": (
            {"Stage1Process": {"progress_bar": True}},
            "show a progress bar during event processing",
        ),
    }

    classes = List(
        [EventSource, CameraCalibrator, ImageDataChecker]
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

        self.event_source = self.add_component(EventSource.from_config(parent=self))

        self.gain_selector = self.add_component(
            GainSelector.from_name(self.gain_selector_type, parent=self)
        )
        self.image_extractor = self.add_component(
            ImageExtractor.from_name(self.image_extractor_type, parent=self)
        )
        self.calibrate = self.add_component(
            CameraCalibrator(
                parent=self,
                image_extractor=self.image_extractor,
                gain_selector=self.gain_selector,
            )
        )
        self.clean = self.add_component(
            ImageCleaner.from_name(self.image_cleaner_type, parent=self)
        )
        self.check_image = self.add_component(ImageDataChecker(parent=self))

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
                camera.to_table().write(
                    self.output_filename,
                    path=f"/configuration/instrument/telescope/camera/{camera}",
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
        """Apply Image Cleaning
       
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

        # apply cleaning

        mask = self.clean(subarray=subarray, image=data.image, tel_id=tel_id)

        clean_image = data.image.copy()
        clean_image[~mask] = 0

        params = ExtendedImageParametersContainer()

        # check if image can be parameterized:

        image_criteria = self.check_image(clean_image)
        self.log.debug(
            "image_criteria: %s",
            list(zip(self.check_image.criteria_names, image_criteria)),
        )

        # parameterize the event if all criteria pass:
        if all(image_criteria):
            params.hillas = hillas_parameters(geom=tel.camera, image=clean_image)
            params.timing = timing_parameters(
                geom=tel.camera,
                image=clean_image,
                pulse_time=data.pulse_time,
                hillas_parameters=params.hillas,
            )
            params.leakage = leakage(
                geom=tel.camera, image=data.image, cleaning_mask=mask
            )
            params.concentration = concentration(
                geom=tel.camera, image=clean_image, hillas_parameters=params.hillas
            )
            params.morphology = morphology(geom=tel.camera, image_mask=mask)
            params.intensity = intensity_statistics(image=clean_image)

        return mask, params

    def _process_events(self, writer):
        self.log.debug("Writing DL1/Event data")
        tel_index = TelEventIndexContainer()
        event_index = EventIndexContainer()
        is_initialized = False

        for event in tqdm(
            self.event_source,
            desc=self.event_source.__class__.__name__,
            total=self.event_source.max_events,
            unit="ev",
            disable=not self.progress_bar,
        ):

            self.log.log(9, "Writing event_id=%s", event.dl0.event_id)

            self.calibrate(event)

            event.mc.prefix = "mc"
            event.trig.prefix = ""
            event_index.event_id = event.index.event_id
            event_index.obs_id = event.index.obs_id
            tel_index.event_id = event.index.event_id
            tel_index.obs_id = event.index.obs_id
            self._cur_obs_id = event.index.obs_id

            # On the first event, we now have a subarray loaded, and other info, so
            # we can write the configuration data.
            if event.count == 0:
                tel_list_transform = create_tel_id_to_tel_index_transform(
                    event.inst.subarray
                )
                writer.add_column_transform(
                    table_name="dl1/event/subarray/trigger",
                    col_name="tels_with_trigger",
                    transform=tel_list_transform,
                )
                self.subarray = event.inst.subarray
                event.inst.subarray.info(printer=self.log.debug)
                self._write_simulation_configuration(writer, event)
                self._write_instrument_configuration(event.inst.subarray)
                is_initialized = True

            # write the subarray tables
            writer.write(
                table_name="dl1/event/subarray/mc_shower",
                containers=[event_index, event.mc],
            )
            writer.write(
                table_name="dl1/event/subarray/trigger",
                containers=[event_index, event.trig],
            )
            # write the telescope tables
            self._write_telescope_event(writer, event, tel_index)

        if is_initialized is False:
            raise ValueError(f"No events found in file: {self.event_source.input_url}")

    def _write_telescope_event(self, writer, event, tel_index):
        """
        add entries to the event/telescope tables for each telescope in a single
        event
        """

        # write the telescope tables
        for tel_id, data in event.dl1.tel.items():

            data.prefix = ""  # don't want a prefix for this container
            telescope = event.inst.subarray.tel[tel_id]
            tel_type = str(telescope)
            tel_index.tel_id = np.int16(tel_id)
            tel_index.tel_type_id = tel_type_string_to_int(tel_type)

            extra = ExtraImageContainer(
                mc_photo_electron_image=event.mc.tel[tel_id].photo_electron_image,
                selected_gain_channel=event.r1.tel[tel_id].selected_gain_channel,
                image_mask=None,  # added later, if computed only
            )

            if self.write_parameters:

                image_mask, params = self._parameterize_image(
                    event.inst.subarray, data, tel_id=tel_id
                )

                self.log.debug("params: %s", params.as_dict(recursive=True))
                self.log.debug("Writing! ******")

                containers_to_write = [
                    tel_index,
                    params.hillas,
                    params.timing,
                    params.leakage,
                    params.concentration,
                    params.morphology,
                    params.intensity,
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
                        table_name=f"dl1/event/telescope/parameters/{tel_type}",
                        containers=containers_to_write,
                    )
                extra.image_mask = image_mask

            if self.write_images:
                # note that we always write the image, even if the image quality
                # criteria are not met (those are only to determine if the parameters
                # can be computed).
                writer.write(
                    table_name=f"dl1/event/telescope/images/{tel_type}",
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

    def _generate_indices(self,):
        with tables.open_file(self.output_filename, mode="a") as h5file:
            if self.write_images:
                self._generate_table_indices(h5file, "/dl1/event/telescope/images")
            self._generate_table_indices(h5file, "/dl1/event/subarray")

    def start(self):

        with HDF5TableWriter(
            self.output_filename, mode="a", add_prefix=True, filters=self._hdf5_filters
        ) as writer:

            if self.write_parameters is False:
                # don't need to write out the image mask if no parameters are computed,
                # since we don't do image cleaning in that case.
                writer.exclude("/dl1/event/telescope/images", "image_mask")

            self._process_events(writer)
            self._write_simulation_histograms(writer)
            self._write_processing_statistics()

        if self.write_index_tables:
            self._generate_indices()

        write_core_provenance(
            output_filename=self.output_filename,
            subarray=self.subarray,
            obs_id=self._cur_obs_id,
        )

    def finish(self):
        pass


def main():
    tool = Stage1Process()
    tool.run()


if __name__ == "__main__":
    main()
