"""
Generate DL1 (a or b) output files in HDF5 format from {R0,R1,DL0} inputs.
"""
# pylint: disable=W0201
import sys
from tqdm.auto import tqdm

from ..calib import CameraCalibrator, GainSelector
from ..core import QualityQuery, Tool
from ..core.traits import Bool, classes_with_traits, flag
from ..image import ImageCleaner, ImageProcessor, ImageModifier
from ..image.extractor import ImageExtractor
from ..io import DataLevel, DataWriter, EventSource, SimTelEventSource, write_table
from ..io.datawriter import DATA_MODEL_VERSION
from ..reco import ShowerProcessor
from ..utils import EventTypeFilter


COMPATIBLE_DATALEVELS = [
    DataLevel.R1,
    DataLevel.DL0,
    DataLevel.DL1_IMAGES,
    DataLevel.DL1_PARAMETERS,
]

__all__ = ["ProcessorTool"]


class ProcessorTool(Tool):
    """
    Process data from lower-data levels up to DL1, including both image
    extraction and optinally image parameterization
    """

    name = "ctapipe-process"
    description = (
        __doc__ + f" This currently uses data model version {DATA_MODEL_VERSION}"
    )
    examples = """
    To process data with all default values:
    > ctapipe-process --input events.simtel.gz --output events.dl1.h5 --progress

    Or use an external configuration file, where you can specify all options:
    > ctapipe-process --config stage1_config.json --progress

    The config file should be in JSON or python format (see traitlets docs). For an
    example, see ctapipe/examples/stage1_config.json in the main code repo.
    """

    progress_bar = Bool(
        help="show progress bar during processing", default_value=False
    ).tag(config=True)

    force_recompute_dl1 = Bool(
        help="Enforce dl1 recomputation even if already present in the input file",
        default_value=False,
    ).tag(config=True)

    force_recompute_dl2 = Bool(
        help="Enforce dl2 recomputation even if already present in the input file",
        default_value=False,
    ).tag(config=True)

    aliases = {
        ("i", "input"): "EventSource.input_url",
        ("o", "output"): "DataWriter.output_path",
        ("t", "allowed-tels"): "EventSource.allowed_tels",
        ("m", "max-events"): "EventSource.max_events",
        "image-cleaner-type": "ImageProcessor.image_cleaner_type",
    }

    flags = {
        "f": (
            {"DataWriter": {"overwrite": True}},
            "Overwrite output file if it exists",
        ),
        **flag(
            "overwrite",
            "DataWriter.overwrite",
            "Overwrite output file if it exists",
            "Don't overwrite output file if it exists",
        ),
        **flag(
            "progress",
            "ProcessorTool.progress_bar",
            "show a progress bar during event processing",
            "don't show a progress bar during event processing",
        ),
        **flag(
            "recompute-dl1",
            "ProcessorTool.force_recompute_dl1",
            "Enforce DL1 recomputation even if already present in the input file",
            "Only compute DL1 if there are no DL1b parameters in the file",
        ),
        **flag(
            "recompute-dl2",
            "ProcessorTool.force_recompute_dl2",
            "Enforce DL2 recomputation even if already present in the input file",
            "Only compute DL2 if there is no shower reconstruction in the file",
        ),
        **flag(
            "write-images",
            "DataWriter.write_images",
            "store DL1/Event/Telescope images in output",
            "don't store DL1/Event/Telescope images in output",
        ),
        **flag(
            "write-parameters",
            "DataWriter.write_parameters",
            "store DL1/Event/Telescope parameters in output",
            "don't store DL1/Event/Telescope parameters in output",
        ),
        **flag(
            "write-stereo-shower",
            "DataWriter.write_stereo_shower",
            "store DL2/Event/Subarray parameters in output",
            "don't DL2/Event/Subarray parameters in output",
        ),
        **flag(
            "write-mono-shower",
            "DataWriter.write_mono_shower",
            "store DL2/Event/Telescope parameters in output",
            "don't store DL2/Event/Telescope parameters in output",
        ),
        **flag(
            "write-index-tables",
            "DataWriter.write_index_tables",
            "generate PyTables index tables for the parameter and image datasets",
        ),
        "camera-frame": (
            {"ImageProcessor": {"use_telescope_frame": False}},
            "Use camera frame for image parameters instead of telescope frame",
        ),
    }

    classes = (
        [CameraCalibrator, DataWriter, ImageProcessor, ShowerProcessor]
        + classes_with_traits(EventSource)
        + classes_with_traits(ImageCleaner)
        + classes_with_traits(ImageExtractor)
        + classes_with_traits(GainSelector)
        + classes_with_traits(QualityQuery)
        + classes_with_traits(ImageModifier)
        + classes_with_traits(EventTypeFilter)
    )

    def setup(self):

        # setup components:
        self.event_source = EventSource(parent=self)
        if not self.event_source.has_any_datalevel(COMPATIBLE_DATALEVELS):
            self.log.critical(
                "%s  needs the EventSource to provide either R1 or DL0 or DL1A data"
                ", %s provides only %s",
                self.name,
                self.event_source,
                self.event_source.datalevels,
            )
            sys.exit(1)

        self.calibrate = CameraCalibrator(
            parent=self, subarray=self.event_source.subarray
        )
        self.process_images = ImageProcessor(
            subarray=self.event_source.subarray, parent=self
        )
        self.process_shower = ShowerProcessor(
            subarray=self.event_source.subarray, parent=self
        )
        self.write = DataWriter(event_source=self.event_source, parent=self)
        self.event_type_filter = EventTypeFilter(parent=self)

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

    @property
    def should_compute_dl2(self):
        """ returns true if we should compute DL2 info """
        if self.force_recompute_dl2:
            return True
        return self.write.write_stereo_shower or self.write.write_mono_shower

    @property
    def should_compute_dl1(self):
        """returns true if we should compute DL1 info"""
        if self.force_recompute_dl1:
            return True

        if DataLevel.DL1_PARAMETERS in self.event_source.datalevels:
            return False

        return self.write.write_parameters or self.should_compute_dl2

    @property
    def should_calibrate(self):
        if self.force_recompute_dl1:
            True

        if (
            self.write.write_images
            and DataLevel.DL1_IMAGES not in self.event_source.datalevels
        ):
            return True

        if self.should_compute_dl1:
            return DataLevel.DL1_IMAGES not in self.event_source.datalevels

        return False

    def _write_processing_statistics(self):
        """write out the event selection stats, etc."""
        # NOTE: don't remove this, not part of DataWriter

        if self.should_compute_dl1:
            image_stats = self.process_images.check_image.to_table(functions=True)
            write_table(
                image_stats,
                self.write.output_path,
                path="/dl1/service/image_statistics",
                append=True,
            )

        if self.should_compute_dl2:
            shower_stats = self.process_shower.check_shower.to_table(functions=True)
            write_table(
                shower_stats,
                self.write.output_path,
                "/dl2/service/image_statistics",
                append=True,
            )

    def start(self):
        """
        Process events
        """
        self.log.info("(re)compute DL1: %s", self.should_compute_dl1)
        self.log.info("(re)compute DL2: %s", self.should_compute_dl2)
        self.event_source.subarray.info(printer=self.log.info)

        for event in tqdm(
            self.event_source,
            desc=self.event_source.__class__.__name__,
            total=self.event_source.max_events,
            unit="ev",
            disable=not self.progress_bar,
        ):

            self.log.debug("Processessing event_id=%s", event.index.event_id)

            if not self.event_type_filter(event):
                continue

            if self.should_calibrate:
                self.calibrate(event)

            if self.should_compute_dl1:
                self.process_images(event)

            if self.should_compute_dl2:
                self.process_shower(event)

            self.write(event)

    def finish(self):
        """
        Last steps after processing events.
        """
        self.write.write_simulation_histograms(self.event_source)
        self.write.finish()
        self.event_source.close()
        self._write_processing_statistics()


def main():
    """ run the tool"""
    tool = ProcessorTool()
    tool.run()


if __name__ == "__main__":
    main()
