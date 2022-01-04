"""
High level processing of showers.

This processor will be able to process a shower/event in 3 steps:
- shower geometry
- estimation of energy (optional, currently unavailable)
- estimation of classification (optional, currently unavailable)

"""
from ctapipe.core import Component, QualityQuery
from ctapipe.core.traits import List
from ctapipe.containers import ArrayEventContainer, ReconstructedGeometryContainer
from ctapipe.instrument import SubarrayDescription
from ctapipe.reco import HillasReconstructor


DEFAULT_SHOWER_PARAMETERS = ReconstructedGeometryContainer(tel_ids=[])


class ShowerQualityQuery(QualityQuery):
    """Configuring shower-wise data checks."""

    quality_criteria = List(
        default_value=[
            ("> 50 phe", "lambda p: p.hillas.intensity > 50"),
            ("Positive width", "lambda p: p.hillas.width.value > 0"),
            ("> 3 pixels", "lambda p: p.morphology.num_pixels > 3"),
        ],
        help=QualityQuery.quality_criteria.help,
    ).tag(config=True)


class ShowerProcessor(Component):
    """
    Needs DL1_PARAMETERS as input.
    Should be run after ImageProcessor which produces such information.

    For the moment it only supports the reconstruction of the shower geometry
    using ctapipe.reco.HillasReconstructor.

    It is planned to support also energy reconstruction and particle type
    classification.
    """

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

        super().__init__(config=config, parent=parent, **kwargs)
        self.subarray = subarray
        self.check_shower = ShowerQualityQuery(parent=self)
        self.reconstructor = HillasReconstructor(self.subarray)

    def reconstruct_geometry(self, event, default=DEFAULT_SHOWER_PARAMETERS):
        """Perform shower reconstruction.

        Parameters
        ----------
        event : container
            A ``ctapipe`` event container
        default: container
            The default 'ReconstructedGeometryContainer' which is
            filled with NaNs.
        Returns
        -------
        ReconstructedGeometryContainer:
            direction in the sky with uncertainty,
            core position on the ground with uncertainty,
            h_max with uncertainty,
            is_valid boolean for successfull reconstruction,
            average intensity of the intensities used for reconstruction,
            measure of algorithm success (if fit),
            list of tel_ids used if stereo, or None if Mono
        """

        # Select only images which pass the shower quality criteria
        hillas_dict = {
            tel_id: dl1.parameters.hillas
            for tel_id, dl1 in event.dl1.tel.items()
            if all(self.check_shower(dl1.parameters))
        }
        self.log.debug("shower_criteria:\n %s", self.check_shower)

        # Reconstruct the shower only if all shower criteria are met
        if len(hillas_dict) > 2:

            self.reconstructor(event)

        else:
            self.log.debug(
                """Less than 2 images passed the quality cuts.
                Returning default ReconstructedGeometryContainer container"""
            )
            event.dl2.stereo.geometry["HillasReconstructor"] = default

    def process_shower_geometry(self, event: ArrayEventContainer):
        """Record the reconstructed shower geometry into the ArrayEventContainer."""

        self.reconstruct_geometry(event)

        self.log.debug(
            "shower geometry:\n %s", event.dl2.stereo.geometry["HillasReconstructor"]
        )

    def __call__(self, event: ArrayEventContainer):
        """
        Perform the full shower geometry reconstruction on the input event.

        Afterwards, optionally perform energy estimation and/or particle
        classification (currently these two operations are not yet supported).

        Parameters
        ----------
        event : ctapipe.containers.ArrayEventContainer
            Top-level container for all event information.
        """

        # This is always done when calling the ShowerProcessor
        self.process_shower_geometry(event)
