"""
Generate DL1 (a or b) output files in HDF5 format from {R0,R1,DL0} inputs.
"""
import sys

from tqdm.autonotebook import tqdm

from ..calib.camera import CameraCalibrator, GainSelector
from ..containers import (
    ImageParametersContainer,
    IntensityStatisticsContainer,
    PeakTimeStatisticsContainer,
    TimingParametersContainer,
)
from ..core import QualityQuery, Tool
from ..core.traits import Bool, List, classes_with_traits, create_class_enum_trait
from ..image import ImageCleaner
from ..image import concentration as concentration_parameters
from ..image import descriptive_statistics, hillas_parameters
from ..image import leakage as leakage_parameters
from ..image import morphology_parameters, timing_parameters
from ..image.extractor import ImageExtractor
from ..io import DataLevel, DL1Writer, EventSource, SimTelEventSource
from ..io.dl1writer import DL1_DATA_MODEL_VERSION


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

    image_cleaner_type = create_class_enum_trait(
        base_class=ImageCleaner, default_value="TailcutsImageCleaner"
    )

    progress_bar = Bool(help="show progress bar during processing").tag(config=True)

    aliases = {
        "input": "EventSource.input_url",
        "output": "DL1Writer.output_path",
        "allowed-tels": "EventSource.allowed_tels",
        "max-events": "EventSource.max_events",
        "image-cleaner-type": "Stage1ProcessorTool.image_cleaner_type",
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
            {"DL1Wroter": {"write_index_tables": True}},
            "generate PyTables index tables for the parameter and image datasets",
        ),
        "overwrite": (
            {"DL1Writer": {"overwrite": True}},
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
        + classes_with_traits(DL1Writer)
    )

    def setup(self):

        # setup components:
        self.event_source = EventSource.from_config(parent=self)

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
        self.clean = ImageCleaner.from_name(
            self.image_cleaner_type, parent=self, subarray=self.event_source.subarray
        )
        self.check_image = ImageQualityQuery(parent=self)
        self._write_dl1 = DL1Writer(event_source=self.event_source, parent=self)

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
        image_stats = self.check_image.to_table(functions=True)
        image_stats.write(
            self._write_dl1.output_path,
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

    def _process_events(self):
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
            self._process_telescope_event(event)
            self._write_dl1(event)

    def _process_telescope_event(self, event):
        """
        Loop over telescopes and process the calibrated images into parameters
        """

        for tel_id, dl1_camera in event.dl1.tel.items():

            if self._write_dl1.write_parameters:
                # compute image parameters only if requested to write them
                dl1_camera.image_mask = self.clean(
                    tel_id=tel_id,
                    image=dl1_camera.image,
                    arrival_times=dl1_camera.peak_time,
                )

                dl1_camera.parameters = self._parameterize_image(
                    tel_id=tel_id,
                    image=dl1_camera.image,
                    signal_pixels=dl1_camera.image_mask,
                    peak_time=dl1_camera.peak_time,
                )

                self.log.debug(
                    "params: %s", dl1_camera.parameters.as_dict(recursive=True)
                )

                if (
                    self.event_source.is_simulation
                    and event.simulation.tel[tel_id].true_image is not None
                ):
                    sim_camera = event.simulation.tel[tel_id]
                    sim_camera.true_parameters = self._parameterize_image(
                        tel_id,
                        image=sim_camera.true_image,
                        signal_pixels=sim_camera.true_image > 0,
                        peak_time=None,  # true image from simulation has no peak time
                    )
                    self.log.debug(
                        "sim params: %s",
                        event.simulation.tel[tel_id].true_parameters.as_dict(
                            recursive=True
                        ),
                    )

    def start(self):
        self._process_events()

    def finish(self):
        self._write_dl1.write_simulation_histograms(self.event_source)
        self._write_dl1.finish()
        self._write_processing_statistics()


def main():
    tool = Stage1ProcessorTool()
    tool.run()


if __name__ == "__main__":
    main()
