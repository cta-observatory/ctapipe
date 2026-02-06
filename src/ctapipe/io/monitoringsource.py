"""
Handles reading of monitoring files
"""

from abc import abstractmethod

import astropy.table
import astropy.time

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
        self.metadata = {"is_simulation": False}

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
    def get_table(
        self,
        monitoring_type: MonitoringType,
        tel_id: int = None,
        **kwargs,
    ) -> astropy.table.Table:
        """
        Get the raw monitoring table for a given monitoring type.

        Parameters
        ----------
        monitoring_type : MonitoringType
            The type of monitoring data to retrieve.
        tel_id : int, optional
            Telescope ID for telescope-level monitoring (camera, telescope pointing).
            None for array-level monitoring (weather, FRAM, LiDAR).
        **kwargs
            Implementation-specific parameters (e.g., subtype for PIXEL_STATISTICS).

        Returns
        -------
        astropy.table.Table
            The monitoring table.

        Raises
        ------
        KeyError
            If monitoring_type is not available.
        TypeError
            If tel_id scope doesn't match monitoring_type requirements.
        """

    @abstractmethod
    def get_values(
        self,
        monitoring_type: MonitoringType,
        time: astropy.time.Time,
        tel_id: int = None,
        **kwargs,
    ):
        """
        Get monitoring values for specific timestamp(s).

        Performs interpolation or nearest-neighbor lookup as appropriate.

        Parameters
        ----------
        monitoring_type : MonitoringType
            The type of monitoring data to retrieve.
        time : astropy.time.Time
            Target timestamp(s). Can be scalar or array.
        tel_id : int, optional
            Telescope ID for telescope-level monitoring. None for array-level.
        **kwargs
            Implementation-specific parameters (e.g., timestamp_tolerance, query_method).

        Returns
        -------
        dict[str, astropy.units.Quantity | numpy.ndarray] or astropy.coordinates.SkyCoord
            Monitoring values at requested time(s). Return type depends on monitoring_type.

        Raises
        ------
        KeyError
            If monitoring_type unavailable
        ValueError
            If time out of bounds.
        TypeError
            If tel_id scope doesn't match monitoring_type requirements.
        """

    @abstractmethod
    def fill_monitoring_container(self, event: ArrayEventContainer):
        """
        Fill the monitoring container for a given event.

        Populates event.monitoring with telescope-level and array-level monitoring
        data for the event's trigger time.

        Parameters
        ----------
        event : ArrayEventContainer
            The event to fill. Uses event.trigger.time for data selection.
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
