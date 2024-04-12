"""
High level processing of showers.
"""
from ..containers import ArrayEventContainer
from ..core import Component, traits
from ..instrument import SubarrayDescription
from .reconstructor import Reconstructor


class ShowerProcessor(Component):
    """
    Run the stereo event reconstruction on the input events.

    This is mainly needed, so that the type of reconstructor can be chosen
    via the configuration system.

    This processor can apply multiple `~ctapipe.reco.Reconstructor` subclasses to
    array events in the event loop.

    This currently includes geometry reconstruction via `~ctapipe.reco.HillasReconstructor`
    or `~ctapipe.reco.HillasIntersection` and machine learning based reconstruction
    of energy and particle type via the reconstructor classes in `~ctapipe.reco`.

    Events must already contain the required inputs. These are dl1 parameters
    for the geometry reconstruction and any feature used by the machine learning
    reconstructors, be it directly as model input or as input to the feature generation.
    This may include previously made dl2 predictions,
    in which case the order of ``reconstructor_types`` is important.
    """

    reconstructor_types = traits.ComponentNameList(
        Reconstructor,
        default_value=["HillasReconstructor"],
        help=(
            "The stereo reconstructors to be used."
            " The reconstructors are applied in the order given,"
            " which is important if e.g. the `~ctapipe.reco.ParticleClassifier`"
            " uses the output of the `~ctapipe.reco.EnergyRegressor` as input."
        ),
    ).tag(config=True)

    def __init__(
        self,
        subarray: SubarrayDescription,
        atmosphere_profile=None,
        config=None,
        parent=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        subarray : SubarrayDescription
            Description of the subarray. Provides information about the
            camera which are useful in calibration. Also required for
            configuring the TelescopeParameter traitlets.
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            This is mutually exclusive with passing a ``parent``.
        parent : ctapipe.core.Component or ctapipe.core.Tool
            Parent of this component in the configuration hierarchy,
            this is mutually exclusive with passing ``config``
        """

        super().__init__(config=config, parent=parent, **kwargs)
        self.subarray = subarray
        self.atmosphere_profile = atmosphere_profile
        if (
            atmosphere_profile is None
            and "ImPACTReconstrcutor" in self.reconstructor_types
        ):
            raise TypeError(
                "Argument 'atmosphere_profile' can not be 'None' if 'ImPACTReconstructor' is in 'reconstructor_types'"
            )
        self.reconstructors = [
            Reconstructor.from_name(
                reco_type,
                subarray=self.subarray,
                atmosphere_profile=atmosphere_profile,
                parent=self,
            )
            for reco_type in self.reconstructor_types
        ]

    def __call__(self, event: ArrayEventContainer):
        """
        Apply all configured stereo reconstructors to the given event.

        Parameters
        ----------
        event : ctapipe.containers.ArrayEventContainer
            Top-level container for all event information.
        """
        for reconstructor in self.reconstructors:
            reconstructor(event)
