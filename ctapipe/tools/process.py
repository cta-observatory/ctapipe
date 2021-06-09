"""
Generate DL1 (a or b) output files in HDF5 format from {R0,R1,DL0} inputs.
"""
import sys
from tqdm.auto import tqdm

from ..calib import CameraCalibrator, GainSelector
from ..core import QualityQuery, Tool
from ..core.traits import Bool, classes_with_traits, flag
from ..image import ImageCleaner, ImageProcessor
from ..image.extractor import ImageExtractor
from ..io import DataLevel, DataWriter, EventSource, SimTelEventSource
from ..io.datawriter import DATA_MODEL_VERSION
from ..reco import ShowerProcessor


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

    progress_bar = Bool(help="show progress bar during processing").tag(config=True)

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
    }

    classes = (
        [CameraCalibrator, DataWriter, ImageProcessor, ShowerProcessor]
        + classes_with_traits(EventSource)
        + classes_with_traits(ImageCleaner)
        + classes_with_traits(ImageExtractor)
        + classes_with_traits(GainSelector)
        + classes_with_traits(QualityQuery)
    )

    def setup(self):

        # setup components:
        self.event_source = EventSource(parent=self)
        compatible_datalevels = [DataLevel.R1, DataLevel.DL0, DataLevel.DL1_IMAGES]
        if not self.event_source.has_any_datalevel(compatible_datalevels):
            self.log.critical(
                f"{self.name} needs the EventSource to provide "
                f"either R1 or DL0 or DL1A data"
                f", {self.event_source} provides only {self.event_source.datalevels}"
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
        self.process_shower = ShowerProcessor(
            subarray=self.event_source.subarray, parent=self
        )
        self.write = DataWriter(event_source=self.event_source, parent=self)

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
        return self.write.write_stereo_shower or self.write.write_mono_shower

    @property
    def should_compute_dl1(self):
        """returns true if we should compute DL1 info"""
        return self.write.write_parameters or self.should_compute_dl2

    def _write_processing_statistics(self):
        """write out the event selection stats, etc."""
        # NOTE: don't remove this, not part of DataWriter

        if self.should_compute_dl1:
            image_stats = self.process_images.check_image.to_table(functions=True)
            image_stats.write(
                self.write.output_path,
                path="/dl1/service/image_statistics",
                append=True,
                serialize_meta=True,
            )

        if self.should_compute_dl2:
            shower_stats = self.process_shower.check_shower.to_table(functions=True)
            shower_stats.write(
                self.write.output_path,
                path="/dl2/service/image_statistics",
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

            self.log.debug("Processessing event_id=%s", event.index.event_id)
            self.calibrate(event)

            if self.should_compute_dl1:
                self.process_images(event)

            if self.should_compute_dl2:
                self.process_shower(event)

            self.write(event)

    def finish(self):
        self.write.write_simulation_histograms(self.event_source)
        self.write.finish()
        self._write_processing_statistics()


def main():
    """ run the tool"""
    tool = ProcessorTool()
    tool.run()


if __name__ == "__main__":
    main()
