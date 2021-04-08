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
            ("Positive charge", "lambda hillas: hillas.intensity > 0"),
            ("Positive width", "lambda hillas: hillas.width.value > 0"),
            ("Positive length", "lambda hillas: hillas.length.value > 0"),
            ("Ellipticity > 0.1", "lambda hillas: hillas.width.value / hillas.length.value > 0.1"),
            ("Ellipticity < 0.6", "lambda hillas: hillas.width.value / hillas.length.value < 0.6")
        ],
        help=QualityQuery.quality_criteria.help,
    ).tag(config=True)


class ShowerProcessor(Component):
    """
    Needs DL1_PARAMETERS as input.
    Should be run after ImageProcessor to produce all DL1b information.

    For the moment it only supports the reconstruction of the shower geometry
    using ctapipe.reco.HillasReconstructor.
    """

    def __init__(
        self,
        subarray: SubarrayDescription,
        reconstruct_energy,
        classify,
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

        super().__init__(config=config, parent=parent, **kwargs)
        self.subarray = subarray
        self.check_shower = ShowerQualityQuery(parent=self)
        self.reconstruct_energy = reconstruct_energy
        self.classify = classify
        self.reconstructor = HillasReconstructor()

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

        # Read only valid HillasContainers (minimum condition to continue)
        hillas_dict = {
            tel_id: dl1.parameters.hillas
            for tel_id, dl1 in event.dl1.tel.items()
            if np.isfinite(event.dl1.tel[tel_id].parameters.hillas.intensity)
        }

        # On top of this check if the shower should be considered based
        # on the user's configuration
        shower_criteria = [self.check_shower(hillas_dict[tel_id]) for tel_id in hillas_dict]
        self.log.debug(
            "shower_criteria: %s",
            list(zip(self.check_shower.criteria_names[1:], shower_criteria)),
        )

        # Reconstruct the shower only if all shower criteria are met
        if np.count_nonzero(shower_criteria) > 2:

            array_pointing = SkyCoord(
                az=event.pointing.array_azimuth,
                alt=event.pointing.array_altitude,
                frame=AltAz(),
            )

            telescopes_pointings = {
                tel_id: SkyCoord(
                    alt=event.pointing.tel[tel_id].altitude,
                    az=event.pointing.tel[tel_id].azimuth,
                    frame=AltAz(),
                )
                for tel_id in hillas_dict
            }

            result = self.reconstructor.predict(hillas_dict,
                                                self.subarray,
                                                array_pointing,
                                                telescopes_pointings)

            return result

        else:
            return default

    def _reconstruct_energy(self, event: ArrayEventContainer):
        raise NotImplementedError("TO DO")

    def _classify_particle_type(self, event: ArrayEventContainer):
        raise NotImplementedError("TO DO")

    def _process_reconstructed_energy(self, event: ArrayEventContainer):
        self._reconstruct_energy(event)

    def _process_reconstructed_classification(self, event: ArrayEventContainer):
        self._classify_particle_type(event)

    def _process_shower_geometry(self, event: ArrayEventContainer):
        """Record the reconstructed shower geometry into the ArrayEventContainer."""

        shower_geometry = self._reconstruct_shower(event)

        self.log.debug("shower geometry: %s", shower_geometry.as_dict(recursive=True))

        event.dl2.shower["HillasReconstructor"] = shower_geometry

    def __call__(self, event: ArrayEventContainer):
        """
        Perform the full shower geometry reconstruction on the input event.

        Afterwards, optionally perform energy estimation and/or particle
        classification (currently these two operations are not yet supported).

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        """

        # This is always done when calling the ShowerProcessor
        self._process_shower_geometry(event)

        if self.reconstruct_energy:
            self._process_reconstructed_classification(event)

        if self.classify:
            self._process_reconstructed_energy(event)
