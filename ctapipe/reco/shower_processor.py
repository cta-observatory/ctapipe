"""
High level processing of showers.

This processor will be able to process a shower/event in 3 steps:
- shower geometry
- estimation of energy (optional, currently unavailable)
- estimation of classification (optional, currently unavailable)

"""
import numpy as np

from astropy.coordinates import SkyCoord, AltAz

from ctapipe.core import Component, QualityQuery
from ctapipe.core.traits import List
from ctapipe.containers import ArrayEventContainer, ReconstructedShowerContainer
from ctapipe.instrument import SubarrayDescription
from ctapipe.reco import HillasReconstructor


DEFAULT_SHOWER_PARAMETERS = ReconstructedShowerContainer()


class ShowerQualityQuery(QualityQuery):
    """Configuring shower-wise data checks."""

    quality_criteria = List(
        default_value=[
            ("lambda p: p.hillas.intensity > 0",
             "lambda p: p.hillas.width > 0",
             "lambda p: p.hillas.length > 0",
             "lambda p: p.hillas.width / p.hillas.length > 0.1",
             "lambda p: p.hillas.width / p.hillas.length < 0.6")
        ],
        help=QualityQuery.quality_criteria.help,
    ).tag(config=True)


class ShowerProcessor(Component):
    """
    Takes DL1/parameters data and estimates the shower geometry.
    Should be run after ImageProcessor to produce all DL1b information.

    For now the only supported reconstructor is HillasReconstructor.
    """

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
        self.check_shower = ShowerQualityQuery(parent=self)
        self._is_simulation = is_simulation
        self.reconstructor = HillasReconstructor(subarray=subarray)

        def _reconstruct_shower(
            self,
            event,
            default=DEFAULT_SHOWER_PARAMETERS,
        ) -> ReconstructedShowerContainer:
            """Perform shower reconstruction.

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
            ReconstructedShowerContainer:
                direction in the sky with uncertainty,
                core position on the ground with uncertainty,
                h_max with uncertainty,
                is_valid boolean for successfull reconstruction,
                average intensity of the intensities used for reconstruction,
                measure of algorithm success (if fit),
                list of tel_ids used if stereo, or None if Mono
            """

            # Read only valid HillasContainers (min condition to continue)
            hillas_dict = {
                tel_id: dl1.parameters.hillas
                for tel_id, dl1 in event.dl1.tel.items()
                if np.isfinite(event.dl1.tel[tel_id].parameters.hillas.intensity)
            }

            if len(hillas_dict) < 2:
                return default

            # On top of this check if the shower should be considered based
            # on the user's configuration
            shower_criteria = self.check_shower(hillas_dict)
            self.log.debug(
                "image_criteria: %s",
                list(zip(self.check_shower.criteria_names[1:], shower_criteria)),
            )

            # Reconstruct the shower only if all shower criteria are met
            if all(shower_criteria):

                array_pointing = SkyCoord(
                    az=event.pointing.array_azimuth,
                    alt=event.pointing.array_altitude,
                    frame=AltAz(),
                )

                telescope_pointings = {
                    tel_id: SkyCoord(
                        alt=event.pointing.tel[tel_id].altitude,
                        az=event.pointing.tel[tel_id].azimuth,
                        frame=AltAz(),
                    )
                    for tel_id in event.dl1.tel.keys()
                }

                return self.reconstructor._predict(hillas_dict,
                                                   self.subarray,
                                                   array_pointing,
                                                   telescope_pointings)

            else:
                return default

        def _reconstruct_energy(self, event: ArrayEventContainer):
            raise NotImplementedError("TO DO")

        def _estimate_classification(self, event: ArrayEventContainer):
            raise NotImplementedError("TO DO")

        def __call__(self, event: ArrayEventContainer):
            """
            Perform the full shower geometry reconstruction on the input event.

            Parameters
            ----------
            event : container
                A `ctapipe` event container
            """

            # This is always done when calling the ShowerProcessor
            self._reconstruct_shower(event)

            if self.estimate_energy:
                self._reconstruct_energy(event)

            if self.estimate_classification:
                self._estimate_classification(event)
