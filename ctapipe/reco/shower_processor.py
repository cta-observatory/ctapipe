"""
High level processing of showers.

This processor will be able to process a shower/event in 3 steps:
- shower geometry
- estimation of energy (optional, currently unavailable)
- estimation of classification (optional, currently unavailable)

"""
from ctapipe.core import Component
from ctapipe.core.traits import create_class_enum_trait
from ctapipe.containers import ArrayEventContainer
from ctapipe.instrument import SubarrayDescription
from ctapipe.reco import Reconstructor


class ShowerProcessor(Component):
    """
    Run the stereo event reconstruction on the input events.

    This is mainly needed, so that the type of reconstructor can be chosen
    via the configuration system.

    Input events must already contain dl1 parameters.
    """
    reconstructor_type = create_class_enum_trait(
        Reconstructor, default_value="HillasReconstructor",
        help="The stereo geometry reconstructor to be used",
    )

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
        self.reconstructor = Reconstructor.from_name(
            self.reconstructor_type,
            subarray=self.subarray,
            parent=self,
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
        k = self.reconstructor_type
        event.dl2.stereo.geometry[k] = self.reconstructor(event)
