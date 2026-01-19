"""
Handles reading of monitoring files
"""

from abc import abstractmethod

from ..containers import ArrayEventContainer
from ..core import TelescopeComponent
from .monitoringtypes import MonitoringType

__all__ = ["MonitoringSource"]


class MonitoringSource(TelescopeComponent):
    """
    Parent class for ``MonitoringSource``.

    ``MonitoringSource`` read input files and fill `~ctapipe.containers.ArrayEventContainer`
    instances with corresponding monitoring data based on the event trigger time.

    A new ``MonitoringSource`` should be created for each type of monitoring file read
    into ctapipe, e.g. HDF5 files are read by the `~ctapipe.io.HDF5MonitoringSource`.

    ``MonitoringSource`` provides a common high-level interface for accessing monitoring
    information from different data sources. Creating an ``MonitoringSource`` for a new
    file format or other monitoring source ensures that data can be accessed in a common way,
    regardless of the file format or data origin.

    ``MonitoringSource`` itself is an abstract class, but will create an
    appropriate subclass. An ``MonitoringSource`` can also be created through the
    configuration system, by passing ``config`` or ``parent`` as appropriate.
    E.g. if using ``MonitoringSource`` inside of a ``Tool``, you would do:

    >>> self.monitoring_source = MonitoringSource(parent=self) # doctest: +SKIP

    """

    plugin_entry_point = "ctapipe_monitoring"

    def __init__(self, subarray=None, config=None, parent=None, **kwargs):
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)
        self.metadata = dict(is_simulation=False)

    @property
    @abstractmethod
    def monitoring_types(self) -> tuple[MonitoringType]:
        """
        The monitoring types provided by this monitoring source

        Returns
        -------
        tuple[ctapipe.io.MonitoringType]
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
        self.close()

    def close(self):
        """Close this event source.

        No-op by default. Should be overridden by sources needing a cleanup-step
        """
        pass
