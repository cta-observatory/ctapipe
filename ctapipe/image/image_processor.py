"""
High level image processing  (ImageProcessor Component)
"""


from ..containers import (
    ArrayEventContainer,
    ImageParametersContainer,
    IntensityStatisticsContainer,
    PeakTimeStatisticsContainer,
    TimingParametersContainer,
)
from ..core import QualityQuery, TelescopeComponent
from ..core.traits import List, create_class_enum_trait
from ..instrument import SubarrayDescription
from . import (
    ImageCleaner,
    concentration_parameters,
    descriptive_statistics,
    hillas_parameters,
    leakage_parameters,
    morphology_parameters,
    timing_parameters,
)


class ImageQualityQuery(QualityQuery):
    """ for configuring image-wise data checks """

    quality_criteria = List(
        default_value=[
            ("size_greater_0", "lambda image_selected: image_selected.sum() > 0")
        ],
        help=QualityQuery.quality_criteria.help,
    ).tag(config=True)


class ImageProcessor(TelescopeComponent):
    """
    Takes DL1/Image data and cleans and parametrizes the images into DL1/parameters.
    Should be run after CameraCalibrator to produce all DL1 information.
    """

    image_cleaner_type = create_class_enum_trait(
        base_class=ImageCleaner, default_value="TailcutsImageCleaner"
    )

    def __init__(
        self,
        subarray: SubarrayDescription,
        is_simulation,
        config=None,
        parent=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        subarray: SubarrayDescription
            Description of the subarray. Provides information about the
            camera which are useful in calibration. Also required for
            configuring the TelescopeParameter traitlets.
        is_simulation: bool
            If true, also process simulated images if they exist
        config: traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            This is mutually exclusive with passing a ``parent``.
        parent: ctapipe.core.Component or ctapipe.core.Tool
            Parent of this component in the configuration hierarchy,
            this is mutually exclusive with passing ``config``
        """

        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)
        self.subarray = subarray
        self.clean = ImageCleaner.from_name(
            self.image_cleaner_type, subarray=subarray, parent=self
        )
        self.check_image = ImageQualityQuery(parent=self)
        self._is_simulation = is_simulation

    def __call__(self, event: ArrayEventContainer):
        self._process_telescope_event(event)

    def _parameterize_image(
        self, tel_id, image, signal_pixels, peak_time=None
    ) -> ImageParametersContainer:
        """Apply image cleaning and calculate image features

        Parameters
        ----------
        tel_id: int
            which telescope is being cleaned
        image: np.ndarray
            image to process
        signal_pixels: np.ndarray[bool]
            image mask
        peak_time: np.ndarray
            peak time image

        Returns
        -------
        ImageParametersContainer:
            cleaning mask, parameters
        """

        tel = self.subarray.tel[tel_id]
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

    def _process_telescope_event(self, event):
        """
        Loop over telescopes and process the calibrated images into parameters
        """

        for tel_id, dl1_camera in event.dl1.tel.items():

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

            self.log.debug("params: %s", dl1_camera.parameters.as_dict(recursive=True))

            if (
                self._is_simulation
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
