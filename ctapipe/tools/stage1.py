"""
Generate DL1 (a or b) output files in HDF5 format from {R0,R1,DL0} inputs.
"""
import sys

from tqdm.autonotebook import tqdm

from ..calib.camera import CameraCalibrator, GainSelector
from ..core import Tool
from ..core.traits import Bool, List, classes_with_traits
from ..image import ImageCleaner, ImageProcessor
from ..image.extractor import ImageExtractor
from ..io import DataLevel, DL1Writer, EventSource, SimTelEventSource
from ..io.dl1writer import DL1_DATA_MODEL_VERSION


class Stage1Tool(Tool):
    """
    Process data from lower-data levels up to DL1, including both image
    extraction and optinally image parameterization
    """

    name = "ctapipe-stage1"
    description = __doc__ + f" This currently writes {DL1_DATA_MODEL_VERSION} DL1 data"
    examples = """
    To process data with all default values:
    > ctapipe-stage1 --input events.simtel.gz --output events.dl1.h5 --progress

    Or use an external configuration file, where you can specify all options:
    > ctapipe-stage1 --config stage1_config.json --progress

    The config file should be in JSON or python format (see traitlets docs). For an
    example, see ctapipe/examples/stage1_config.json in the main code repo.
    """

    progress_bar = Bool(help="show progress bar during processing").tag(config=True)

    aliases = {
        "input": "EventSource.input_url",
        "output": "DL1Writer.output_path",
        "allowed-tels": "EventSource.allowed_tels",
        "max-events": "EventSource.max_events",
        "image-cleaner-type": "ImageProcessor.image_cleaner_type",
    }

    flags = {
        "write-images": (
            {"DL1Writer": {"write_images": True}},
            "store DL1/Event/Telescope images in output",
        ),
        "write-parameters": (
            {"DL1Writer": {"write_parameters": True}},
            "store DL1/Event/Telescope parameters in output",
        ),
        "write-index-tables": (
            {"DL1Writer": {"write_index_tables": True}},
            "generate PyTables index tables for the parameter and image datasets",
        ),
        "overwrite": (
            {"DL1Writer": {"overwrite": True}},
            "Overwrite output file if it exists",
        ),
        "progress": (
            {"Stage1Tool": {"progress_bar": True}},
            "show a progress bar during event processing",
        ),
    }

    classes = List(
        [CameraCalibrator, DL1Writer, ImageProcessor]
        + classes_with_traits(EventSource)
        + classes_with_traits(ImageCleaner)
        + classes_with_traits(ImageExtractor)
        + classes_with_traits(GainSelector)
    )

    def setup(self):

        # setup components:
        self.event_source = EventSource(parent=self)

        datalevels = self.event_source.datalevels
        if DataLevel.R1 not in datalevels and DataLevel.DL0 not in datalevels:
            self.log.critical(
                f"{self.name} needs the EventSource to provide either R1 or DL0 data"
                f", {self.event_source} provides only {datalevels}"
            )
            sys.exit(1)

        self.calibrate = CameraCalibrator(
            parent=self, subarray=self.event_source.subarray
        )
        self.process_images = ImageProcessor(
            subarray=self.event_source.subarray,
            is_simulation=self.event_source.is_simulation,
            parent=self,
        )
        self.write_dl1 = DL1Writer(event_source=self.event_source, parent=self)

        # warn if max_events prevents writing the histograms
        if (
            isinstance(self.event_source, SimTelEventSource)
            and self.event_source.max_events
            and self.event_source.max_events > 0
        ):
            self.log.warning(
                "No Simulated shower distributions will be written because "
                "EventSource.max_events is set to a non-zero number (and therefore "
                "shower distributions read from the input Simulation file are invalid)."
            )

    def _write_processing_statistics(self):
        """ write out the event selection stats, etc. """
        # NOTE: don't remove this, not part of DL1Writer
        image_stats = self.process_images.check_image.to_table(functions=True)
        image_stats.write(
            self.write_dl1.output_path,
            path="/dl1/service/image_statistics",
            append=True,
            serialize_meta=True,
        )

    def start(self):
        self.event_source.subarray.info(printer=self.log.info)
        for event in tqdm(
            self.event_source,
            desc=self.event_source.__class__.__name__,
            total=self.event_source.max_events,
            unit="ev",
            disable=not self.progress_bar,
        ):

            self.log.log(9, "Processessing event_id=%s", event.index.event_id)
            self.calibrate(event)
            if self.write_dl1.write_parameters:
                self.process_images(event)
            self.write_dl1(event)

    def finish(self):
        self.write_dl1.write_simulation_histograms(self.event_source)
        self.write_dl1.finish()
        self._write_processing_statistics()


def main():
    """ run the tool"""
    tool = Stage1Tool()
    tool.run()
