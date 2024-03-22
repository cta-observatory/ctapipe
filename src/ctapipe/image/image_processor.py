"""
High level image processing  (ImageProcessor Component)
"""

from copy import deepcopy

import numpy as np

from ctapipe.coordinates import TelescopeFrame

from ..containers import (
    ArrayEventContainer,
    CameraHillasParametersContainer,
    CameraTimingParametersContainer,
    ImageParametersContainer,
    IntensityStatisticsContainer,
    PeakTimeStatisticsContainer,
    TimingParametersContainer,
)
from ..core import QualityQuery, TelescopeComponent
from ..core.traits import Bool, BoolTelescopeParameter, ComponentName, List
from ..instrument import SubarrayDescription
from .cleaning import ImageCleaner
from .concentration import concentration_parameters
from .hillas import hillas_parameters
from .leakage import leakage_parameters
from .modifications import ImageModifier
from .morphology import morphology_parameters
from .statistics import descriptive_statistics
from .timing import timing_parameters

# avoid use of base containers for unparameterized images
DEFAULT_IMAGE_PARAMETERS = ImageParametersContainer()
DEFAULT_TRUE_IMAGE_PARAMETERS = ImageParametersContainer()
DEFAULT_TRUE_IMAGE_PARAMETERS.intensity_statistics = IntensityStatisticsContainer(
    max=np.int32(-1),
    min=np.int32(-1),
    mean=np.float64(np.nan),
    std=np.float64(np.nan),
    skewness=np.float64(np.nan),
    kurtosis=np.float64(np.nan),
)
DEFAULT_TIMING_PARAMETERS = TimingParametersContainer()
DEFAULT_TIMING_PARAMETERS_CAMFRAME = CameraTimingParametersContainer()
DEFAULT_PEAKTIME_STATISTICS = PeakTimeStatisticsContainer()


DEFAULT_IMAGE_PARAMETERS_CAMFRAME = deepcopy(DEFAULT_IMAGE_PARAMETERS)
DEFAULT_IMAGE_PARAMETERS_CAMFRAME.hillas = CameraHillasParametersContainer()
DEFAULT_IMAGE_PARAMETERS_CAMFRAME.timing = CameraTimingParametersContainer()


class ImageQualityQuery(QualityQuery):
    """for configuring image-wise data checks"""

    quality_criteria = List(
        default_value=[("size_greater_0", "image.sum() > 0")],
        help=QualityQuery.quality_criteria.help,
    ).tag(config=True)


class ImageProcessor(TelescopeComponent):
    """
    Takes DL1/Image data and cleans and parametrizes the images into DL1/parameters.
    Should be run after CameraCalibrator to produce all DL1 information.
    """

    image_cleaner_type = ComponentName(
        ImageCleaner, default_value="TailcutsImageCleaner"
    ).tag(config=True)

    use_telescope_frame = Bool(
        default_value=True,
        help="Whether to calculate parameters in the telescope or camera frame",
    ).tag(config=True)

    apply_image_modifier = BoolTelescopeParameter(
        default_value=False, help="If true, apply ImageModifier to dl1 images"
    ).tag(config=True)

    def __init__(
        self, subarray: SubarrayDescription, config=None, parent=None, **kwargs
    ):
        """
        Parameters
        ----------
        subarray: SubarrayDescription
            Description of the subarray. Provides information about the
            camera which are useful in calibration. Also required for
            configuring the TelescopeParameter traitlets.
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
        self.modify = ImageModifier(subarray=subarray, parent=self)

        self.check_image = ImageQualityQuery(parent=self)

        self.default_image_container = DEFAULT_IMAGE_PARAMETERS_CAMFRAME
        if self.use_telescope_frame:
            self.default_image_container = DEFAULT_IMAGE_PARAMETERS
            telescope_frame = TelescopeFrame()
            self.telescope_frame_geometries = {
                tel_id: self.subarray.tel[tel_id].camera.geometry.transform_to(
                    telescope_frame
                )
                for tel_id in self.subarray.tel
            }

    def __call__(self, event: ArrayEventContainer):
        self._process_telescope_event(event)

    def _parameterize_image(
        self,
        tel_id,
        image,
        signal_pixels,
        peak_time=None,
        default=DEFAULT_IMAGE_PARAMETERS,
    ) -> ImageParametersContainer:
        """
        Apply image cleaning and calculate image features

        Parameters
        ----------
        tel_id : int
            which telescope is being cleaned
        image : np.ndarray
            image to process
        signal_pixels : np.ndarray[bool]
            image mask
        peak_time : np.ndarray
            peak time image

        Returns
        -------
        parameters : ImageParametersContainer
        """
        image_selected = image[signal_pixels]

        if self.use_telescope_frame:
            # Use the transformed geometries
            geometry = self.telescope_frame_geometries[tel_id]
        else:
            geometry = self.subarray.tel[tel_id].camera.geometry

        # check if image can be parameterized:
        image_criteria = self.check_image(image=image_selected)
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
                if self.use_telescope_frame:
                    timing = DEFAULT_TIMING_PARAMETERS
                else:
                    timing = DEFAULT_TIMING_PARAMETERS_CAMFRAME

                peak_time_statistics = DEFAULT_PEAKTIME_STATISTICS

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
        return default

    def _process_telescope_event(self, event):
        """
        Loop over telescopes and process the calibrated images into parameters
        """
        for tel_id, dl1_camera in event.dl1.tel.items():
            if self.apply_image_modifier.tel[tel_id]:
                dl1_camera.image = self.modify(tel_id=tel_id, image=dl1_camera.image)

            dl1_camera.image_mask = self.clean(
                tel_id=tel_id,
                image=dl1_camera.image,
                arrival_times=dl1_camera.peak_time,
                monitoring=event.mon.tel[tel_id],
            )

            dl1_camera.parameters = self._parameterize_image(
                tel_id=tel_id,
                image=dl1_camera.image,
                signal_pixels=dl1_camera.image_mask,
                peak_time=dl1_camera.peak_time,
                default=self.default_image_container,
            )

            self.log.debug("params: %s", dl1_camera.parameters.as_dict(recursive=True))

            if (
                event.simulation is not None
                and tel_id in event.simulation.tel
                and event.simulation.tel[tel_id].true_image is not None
            ):
                sim_camera = event.simulation.tel[tel_id]
                sim_camera.true_parameters = self._parameterize_image(
                    tel_id,
                    image=sim_camera.true_image,
                    signal_pixels=sim_camera.true_image > 0,
                    peak_time=None,  # true image from simulation has no peak time
                    default=DEFAULT_TRUE_IMAGE_PARAMETERS,
                )
                for container in sim_camera.true_parameters.values():
                    if not container.prefix.startswith("true_"):
                        container.prefix = f"true_{container.prefix}"

                self.log.debug(
                    "sim params: %s",
                    event.simulation.tel[tel_id].true_parameters.as_dict(
                        recursive=True
                    ),
                )
