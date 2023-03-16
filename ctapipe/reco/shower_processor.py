"""
High level processing of showers.
"""
from ..containers import ArrayEventContainer
from ..core import Component, traits
from ..instrument import SubarrayDescription
from .reconstructor import Reconstructor
from .impact_distance import shower_impact_distance
from ctapipe.core import traits

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

    advanced_reconstructor_type = traits.CaselessStrEnum(["ImPACTReconstructor", ""], default_value="", help="name minimiser to use in the fit").tag(config=True)

    def __init__(
        self, subarray: SubarrayDescription, config=None, parent=None, **kwargs
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

        self.reconstructor = Reconstructor.from_name(
            self.reconstructor_type,
            subarray=self.subarray,
            parent=self,
        )
        if self.advanced_reconstructor_type != "":
            self.advanced_reconstructor = Reconstructor.from_name(self.advanced_reconstructor_type,
            subarray=self.subarray, parent=self)


    def __call__(self, event: ArrayEventContainer):
        """
        Apply all configured stereo reconstructors to the given event.

        Parameters
        ----------
        event : ctapipe.containers.ArrayEventContainer
            Top-level container for all event information.
        """

        k = self.reconstructor_type            
        event.dl2.stereo.geometry[k] = self.reconstructor(event)
        
        if self.advanced_reconstructor_type != "":
            geometry, energy = self.advanced_reconstructor(event)
            event.dl2.stereo.geometry[self.advanced_reconstructor_type] = geometry
            event.dl2.stereo.energy[self.advanced_reconstructor_type] = energy

        # compute and store the impact parameter for each reconstruction (for
        # now there is only one, but in the future this should be a loop over
        # reconstructors)

        # for the stereo reconstructor:
        impact_distances = shower_impact_distance(
            shower_geom=event.dl2.stereo.geometry[k], subarray=self.subarray
        )

        for tel_id in event.trigger.tels_with_trigger:
            tel_index = self.subarray.tel_indices[tel_id]
            event.dl2.tel[tel_id].impact[k] = TelescopeImpactParameterContainer(
                distance=impact_distances[tel_index],
                prefix=f"{self.reconstructor_type}_tel",
            )

