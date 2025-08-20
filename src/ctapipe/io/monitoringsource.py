"""
Handles reading of monitoring files
"""
from abc import abstractmethod

from ..containers import ArrayEventContainer
from ..core import TelescopeComponent
from ..core.traits import Undefined
from .monitoringtypes import MonitoringTypes

__all__ = ["MonitoringSource"]


class MonitoringSource(TelescopeComponent):
    """
    Parent class for MonitoringSource.

    MonitoringSources read input files and fill `~ctapipe.containers.ArrayEventContainer`
    instances with monitoring data.

    Parameters
    ----------
    input_url : str | Path
        Path to the input monitoring file.
    """

    plugin_entry_point = "ctapipe_monitoring"

    def __init__(
        self, subarray=None, input_url=None, config=None, parent=None, **kwargs
    ):
        # traitlets differentiates between not getting the kwarg
        # and getting the kwarg with a None value.
        # the latter overrides the value in the config with None, the former
        # enables getting it from the config.
        if input_url not in {None, Undefined}:
            kwargs["input_url"] = input_url

        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)
        self.metadata = dict(is_simulation=False)
        self.log.info(f"INPUT PATH = {self.input_url}")

    @property
    @abstractmethod
    def monitoring_types(self) -> tuple[MonitoringTypes]:
        """
        The monitoring types provided by this monitoring source

        Returns
        -------
        tuple[ctapipe.io.MonitoringTypes]
        """

    def has_any_monitoring_types(self, monitoring_types) -> bool:
        """
        Check if any of `monitoring_types` is in self.monitoring_types

        Parameters
        ----------
        monitoring_types: Iterable
            Iterable of monitoring types
        """
        return any(mt in self.monitoring_types for mt in monitoring_types)

    @abstractmethod
    def fill_monitoring_container(self, event: ArrayEventContainer):
        """
        Fill the monitoring container for a given event.

        Parameters
        ----------
        event : ArrayEventContainer
            The event to fill the monitoring container for.
        """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def close(self):
        """Close this event source.

        No-op by default. Should be overridden by sources needing a cleanup-step
        """
        pass
