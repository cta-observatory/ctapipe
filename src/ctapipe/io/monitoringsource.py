"""
Handles reading of monitoring files
"""

from abc import abstractmethod
from typing import Any

import astropy.table
import astropy.time

from ..containers import ArrayEventContainer
from ..core import TelescopeComponent
from .monitoringtypes import MonitoringType, TelescopeMonitoringType

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

    @property
    @abstractmethod
    def available_data(self) -> tuple[tuple[MonitoringType, str | None], ...]:
        """
        Returns the available monitoring types of this source.

        Returns
        -------
        available_data : tuple[tuple[ctapipe.io.MonitoringType, str | None]]
            A tuple of (type, subtype) pairs of the available monitoring data.
        """

    @property
    @abstractmethod
    def available_telescope_data(
        self,
    ) -> dict[int, tuple[TelescopeMonitoringType, str | None]]:
        """
        Returns the available telescope monitoring types of this source.

        Returns
        -------
        available_data : dict[int, tuple[ctapipe.io.TelescopeMonitoringType, str | None]]
            A dict mapping tel_id to (type, subtype) pairs of the available monitoring data.
        """

    @abstractmethod
    def get_telescope_table(
        self,
        tel_id: int,
        monitoring_type: TelescopeMonitoringType,
        subtype: str | None = None,
    ):
        """
        Get the monitoring table for a given telescope monitoring type.

        Parameters
        ----------
        monitoring_type : MonitoringType
            The type of monitoring data to retrieve.
        subtype : str | None
            Optional subtype, e.g. for PIXEL_STATISTICS type

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
    def get_table(
        self,
        monitoring_type: MonitoringType,
        subtype: str | None = None,
    ) -> astropy.table.Table:
        """
        Get the monitoring table for a given monitoring type.

        Parameters
        ----------
        monitoring_type : MonitoringType
            The type of monitoring data to retrieve.
        subtype : str | None
            Optional subtype

        Returns
        -------
        astropy.table.Table
            The monitoring table.

        Raises
        ------
        KeyError
            If monitoring_type / subtype is not available.
        """

    @abstractmethod
    def get_values(
        self,
        time: astropy.time.Time,
        monitoring_type: MonitoringType,
        subtype: str | None = None,
    ) -> Any:
        """
        Get monitoring values for specific timestamp(s).

        Performs interpolation or nearest-neighbor lookup as appropriate.

        Parameters
        ----------
        time : astropy.time.Time
            Target timestamp(s). Can be scalar or array.
        monitoring_type : MonitoringType
            The type of monitoring data to retrieve.
        subtype : str | None
            Optional subtype

        Returns
        -------
        monitoring_data :
            Monitoring values at requested time(s). Return type depends on monitoring_type.

        Raises
        ------
        KeyError
            If monitoring_type unavailable
        ValueError
            If time out of bounds.
        """

    @abstractmethod
    def get_telescope_values(
        self,
        tel_id: int,
        time: astropy.time.Time,
        monitoring_type: TelescopeMonitoringType,
        subtype: str | None = None,
    ) -> Any:
        """
        Get monitoring values for specific timestamp(s).

        Performs interpolation or nearest-neighbor lookup as appropriate.

        Parameters
        ----------
        time : astropy.time.Time
            Target timestamp(s). Can be scalar or array.
        monitoring_type : MonitoringType
            The type of monitoring data to retrieve.
        subtype : str | None
            Optional subtype

        Returns
        -------
        monitoring_data :
            Monitoring values at requested time(s). Return type depends on monitoring_type.

        Raises
        ------
        KeyError
            If monitoring_type unavailable
        ValueError
            If time out of bounds.
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
