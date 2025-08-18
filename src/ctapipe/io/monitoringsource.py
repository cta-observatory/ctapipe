"""
Handles reading of monitoring files
"""
from abc import abstractmethod
from collections.abc import Generator

import astropy
import tables
from astropy.utils.decorators import lazyproperty

from ..containers import ArrayEventContainer, MonitoringCameraContainer
from ..core.component import TelescopeComponent
from ..core.traits import Path, Undefined
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

    input_url = Path(help="Path to the input file containing monitoring data.").tag(
        config=True
    )

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

    @abstractmethod
    def _get_monitoring_types(file: tables.File | str | Path) -> tuple[MonitoringTypes]:
        """Get the monitoring types present in the monitoring file

        Parameters
        ----------
        file : tables.File | str | Path
            The file to check for monitoring types.

        Returns
        -------
        tuple[MonitoringTypes]
            A tuple of the monitoring types present in the file.
        """

    @lazyproperty
    def monitoring_types(self):
        return self._get_monitoring_types(self.file_)

    def has_any_monitoring_type(self, monitoring_types) -> bool:
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

    @abstractmethod
    def get_camera_monitoring_container(
        self, tel_id: int, time: astropy.time.Time
    ) -> Generator[MonitoringCameraContainer]:
        """
        Get the camera monitoring container for a specific time.

        Parameters
        ----------
        tel_id : int
            The telescope ID to find the camera monitoring data for.
        time : astropy.time.Time
            The target timestamp to find the camera monitoring data for.

        Returns
        -------
        Generator[MonitoringCameraContainer]
            A generator yielding the camera monitoring containers for the specified time.
        """

    def close(self):
        """Close this event source.

        No-op by default. Should be overridden by sources needing a cleanup-step
        """
        pass
