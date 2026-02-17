"""
Generate DL1 (a or b) output files in HDF5 format from {R0,R1,DL0} inputs.
"""

# pylint: disable=W0201
import sys

import astropy.units as u
import numpy as np
from tqdm.auto import tqdm

from ..calib import CameraCalibrator, GainSelector
from ..core import QualityQuery, Tool, ToolConfigurationError
from ..core.traits import Bool, ComponentName, List, classes_with_traits, flag
from ..exceptions import InputMissing
from ..image import ImageCleaner, ImageModifier, ImageProcessor
from ..image.extractor import ImageExtractor
from ..image.muon import MuonProcessor
from ..instrument import SoftwareTrigger
from ..io import (
    DataLevel,
    DataWriter,
    EventSource,
    MonitoringSource,
    MonitoringType,
    metadata,
    write_table,
)
from ..io.datawriter import DATA_MODEL_VERSION
from ..io.hdf5dataformat import (
    DL1_IMAGE_STATISTICS_TABLE,
    DL2_EVENT_STATISTICS_GROUP,
)
from ..reco import Reconstructor, ShowerProcessor
from ..reco.preprocessing import horizontal_to_telescope
from ..utils import EventTypeFilter

COMPATIBLE_DATALEVELS = [
    DataLevel.R1,
    DataLevel.DL0,
    DataLevel.DL1_IMAGES,
    DataLevel.DL1_PARAMETERS,
]

COMPATIBLE_MONITORINGTYPES = [
    MonitoringType.PIXEL_STATISTICS,
    MonitoringType.CAMERA_COEFFICIENTS,
    MonitoringType.TELESCOPE_POINTINGS,
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

    monitoring_source_list = List(
        ComponentName(MonitoringSource),
        help=(
            "List of monitoring sources to use during processing "
            "if the calibration of the data is requested. Later "
            "MonitoringSource instances overwrite earlier ones if "
            "the monitoring types of the different instances overlap."
        ),
        default_value=[],
    ).tag(config=True)

    aliases = {
        ("i", "input"): "EventSource.input_url",
        ("o", "output"): "DataWriter.output_path",
        ("t", "allowed-tels"): "EventSource.allowed_tels",
        ("m", "max-events"): "EventSource.max_events",
        "monitoring-source": "ProcessorTool.monitoring_source_list",
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
        + classes_with_traits(MonitoringSource)
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
        try:
            self.event_source = self.enter_context(EventSource(parent=self))
        except InputMissing:
            self.log.critical(
                "Specifying EventSource.input_url is required (via -i, --input or a config file)."
            )

            self.exit(1)

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

        # Setup the monitoring sources
        self._monitoring_sources = []
        for mon_source_name in self.monitoring_source_list:
            mon_source = self.enter_context(
                MonitoringSource.from_name(
                    mon_source_name, subarray=subarray, parent=self
                )
            )
            # Check if monitoring source has compatible monitoring types
            if not mon_source.has_any_monitoring_types(COMPATIBLE_MONITORINGTYPES):
                msg = (
                    f"'{mon_source_name}' needs the MonitoringSource to provide at least "
                    f"one of these monitoring types: {COMPATIBLE_MONITORINGTYPES}, "
                    f"{mon_source_name} provides only '{mon_source.monitoring_types}'. "
                    f"Please make sure the '{mon_source_name}' and its input "
                    f"are suitable for calibrating the data you are processing."
                )
                self.log.critical(msg)
                raise ToolConfigurationError(msg)
            # Append the monitoring source to the list if it has compatible monitoring types
            self._monitoring_sources.append(mon_source)

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
                path=DL1_IMAGE_STATISTICS_TABLE,
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
                    f"{DL2_EVENT_STATISTICS_GROUP}/{reconstructor_name}",
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

            for mon_source in self._monitoring_sources:
                mon_source.fill_monitoring_container(event)

            if self.should_calibrate:
                self.calibrate(event)

            if self.should_compute_dl1:
                self.process_images(event)

                if event.simulation is not None:
                    shower = event.simulation.shower
                    for tel_id, dl1 in event.dl1.tel.items():
                        if (
                            tel_id not in event.simulation.tel
                            or event.simulation.tel[tel_id].true_parameters is None
                        ):
                            continue

                        true_param = event.simulation.tel[tel_id].true_parameters
                        hillas = dl1.parameters.hillas

                        if (
                            hillas is not None
                            and np.isfinite(hillas.fov_lat)
                            and np.isfinite(hillas.fov_lon)
                        ):
                            pointing = event.monitoring.tel[tel_id].pointing
                            # calculate true disp
                            fov_lon, fov_lat = horizontal_to_telescope(
                                alt=shower.alt,
                                az=shower.az,
                                pointing_alt=pointing.altitude,
                                pointing_az=pointing.azimuth,
                            )
                            # numpy's trigonometric functions need radians
                            psi = hillas.psi.to_value(u.rad)
                            cog_lon = hillas.fov_lon
                            cog_lat = hillas.fov_lat

                            delta_lon = fov_lon - cog_lon
                            delta_lat = fov_lat - cog_lat

                            true_disp = (
                                np.cos(psi) * delta_lon + np.sin(psi) * delta_lat
                            )
                            true_param.true_disp.norm = np.abs(true_disp)
                            true_param.true_disp.sign = np.sign(true_disp.value)

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
