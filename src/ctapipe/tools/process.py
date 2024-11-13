"""
Generate DL1 (a or b) output files in HDF5 format from {R0,R1,DL0} inputs.
"""
# pylint: disable=W0201
import sys

from tqdm.auto import tqdm

from ..calib import CameraCalibrator, GainSelector
from ..core import QualityQuery, Tool
from ..core.traits import Bool, classes_with_traits, flag
from ..image import ImageCleaner, ImageModifier, ImageProcessor
from ..image.extractor import ImageExtractor
from ..image.muon import MuonProcessor
from ..instrument import SoftwareTrigger
from ..io import (
    DataLevel,
    DataWriter,
    EventSource,
    metadata,
    write_table,
)
from ..io.datawriter import DATA_MODEL_VERSION
from ..reco import Reconstructor, ShowerProcessor
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
    Process data from lower-data levels up to DL1 and DL2, including image
    extraction and optionally image parameterization as well as muon analysis
    and shower reconstruction.

    Note that the muon analysis and shower reconstruction both depend on
    parametrized images and therefore compute image parameters even if
    DataWriter.write_dl1_parameters=False in case these are not already present
    in the input file.
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
        "reconstructor": "ShowerProcessor.reconstructor_types",
        "image-cleaner-type": "ImageProcessor.image_cleaner_type",
    }

    flags = {
        "overwrite": (
            {"DataWriter": {"overwrite": True}},
            "Overwrite output file if it exists",
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
            "DataWriter.write_dl1_images",
            "store DL1/Event/Telescope images in output",
            "don't store DL1/Event/Telescope images in output",
        ),
        **flag(
            "write-parameters",
            "DataWriter.write_dl1_parameters",
            "store DL1/Event/Telescope parameters in output",
            "don't store DL1/Event/Telescope parameters in output",
        ),
        **flag(
            "write-showers",
            "DataWriter.write_dl2",
            "store DL2/Event parameters in output",
            "don't DL2/Event parameters in output",
        ),
        **flag(
            "write-index-tables",
            "DataWriter.write_index_tables",
            "generate PyTables index tables for the parameter and image datasets",
        ),
        **flag(
            "write-muon-parameters",
            "DataWriter.write_muon_parameters",
            "store DL1/Event/Telescope muon parameters in output",
            "don't store DL1/Event/Telescope muon parameters in output",
        ),
        "camera-frame": (
            {"ImageProcessor": {"use_telescope_frame": False}},
            "Use camera frame for image parameters instead of telescope frame",
        ),
    }

    classes = (
        [
            CameraCalibrator,
            DataWriter,
            ImageProcessor,
            MuonProcessor,
            ShowerProcessor,
            metadata.Instrument,
            metadata.Contact,
            SoftwareTrigger,
        ]
        + classes_with_traits(EventSource)
        + classes_with_traits(ImageCleaner)
        + classes_with_traits(ImageExtractor)
        + classes_with_traits(GainSelector)
        + classes_with_traits(QualityQuery)
        + classes_with_traits(ImageModifier)
        + classes_with_traits(EventTypeFilter)
        + classes_with_traits(Reconstructor)
    )

    def setup(self):
        # setup components:
        self.event_source = self.enter_context(EventSource(parent=self))

        if not self.event_source.has_any_datalevel(COMPATIBLE_DATALEVELS):
            self.log.critical(
                "%s  needs the EventSource to provide at least one of these datalevels: %s"
                ", %s provides only %s",
                self.name,
                COMPATIBLE_DATALEVELS,
                self.event_source,
                self.event_source.datalevels,
            )
            sys.exit(1)

        subarray = self.event_source.subarray
        self.software_trigger = SoftwareTrigger(parent=self, subarray=subarray)
        self.calibrate = CameraCalibrator(parent=self, subarray=subarray)
        self.process_images = ImageProcessor(subarray=subarray, parent=self)
        self.process_shower = ShowerProcessor(
            subarray=subarray,
            atmosphere_profile=self.event_source.atmosphere_density_profile,
            parent=self,
        )
        self.write = self.enter_context(
            DataWriter(event_source=self.event_source, parent=self)
        )

        self.process_muons = None
        if self.should_compute_muon_parameters:
            self.process_muons = MuonProcessor(subarray=subarray, parent=self)

        self.event_type_filter = EventTypeFilter(parent=self)

    @property
    def should_compute_dl2(self):
        """returns true if we should compute DL2 info"""
        if self.force_recompute_dl2:
            return True

        return self.write.write_dl2

    @property
    def should_compute_dl1(self):
        """returns true if we should compute DL1 info"""
        if self.force_recompute_dl1:
            return True

        if DataLevel.DL1_PARAMETERS in self.event_source.datalevels:
            return False

        return (
            self.write.write_dl1_parameters
            or self.should_compute_dl2
            or self.should_compute_muon_parameters
        )

    @property
    def should_calibrate(self):
        """returns true if data should be calibrated"""
        if self.force_recompute_dl1:
            return True

        if (
            self.write.write_dl1_images
            and DataLevel.DL1_IMAGES not in self.event_source.datalevels
        ):
            return True

        if self.should_compute_dl1:
            return DataLevel.DL1_IMAGES not in self.event_source.datalevels

        return False

    @property
    def should_compute_muon_parameters(self):
        """returns true if we should compute muon parameters info"""
        if self.write.write_muon_parameters:
            return True

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
            reconstructors = self.process_shower.reconstructors
            reconstructor_names = self.process_shower.reconstructor_types
            for reconstructor_name, reconstructor in zip(
                reconstructor_names, reconstructors
            ):
                write_table(
                    reconstructor.quality_query.to_table(functions=True),
                    self.write.output_path,
                    f"/dl2/service/tel_event_statistics/{reconstructor_name}",
                    append=True,
                )

    def start(self):
        """
        Process events
        """
        self.log.info("applying calibration: %s", self.should_calibrate)
        self.log.info("(re)compute DL1: %s", self.should_compute_dl1)
        self.log.info("(re)compute DL2: %s", self.should_compute_dl2)
        self.log.info(
            "compute muon parameters: %s", self.should_compute_muon_parameters
        )
        self.event_source.subarray.info(printer=self.log.info)

        for event in tqdm(
            self.event_source,
            desc=self.event_source.__class__.__name__,
            total=self.event_source.max_events,
            unit="ev",
            disable=not self.progress_bar,
        ):
            self.log.debug("Processing event_id=%s", event.index.event_id)
            if not self.event_type_filter(event):
                continue

            if not self.software_trigger(event):
                self.log.debug(
                    "Skipping event %i due to software trigger", event.index.event_id
                )
                continue

            if self.should_calibrate:
                self.calibrate(event)

            if self.should_compute_dl1:
                self.process_images(event)

            if self.should_compute_muon_parameters:
                self.process_muons(event)

            if self.should_compute_dl2:
                self.process_shower(event)

            self.write(event)

    def finish(self):
        """
        Last steps after processing events.
        """
        shower_dists = self.event_source.simulated_shower_distributions
        self.write.write_simulated_shower_distributions(shower_dists)

        self._write_processing_statistics()


def main():
    """run the tool"""
    tool = ProcessorTool()
    tool.run()


if __name__ == "__main__":
    main()
