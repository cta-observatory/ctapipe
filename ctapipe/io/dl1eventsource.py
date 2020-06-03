from ctapipe.io import EventSource
import tables
import matplotlib.pyplot as plt
from ctapipe.io.datalevels import DataLevel

from ctapipe.instrument import (
    TelescopeDescription,
    SubarrayDescription,
    CameraDescription,
    CameraGeometry,
    CameraReadout,
    OpticsDescription,
)
from astropy.table import Table

from ctapipe.containers import EventAndMonDataContainer, EventType
from ctapipe.instrument.camera import UnknownPixelShapeWarning
from ctapipe.instrument.guess import guess_telescope, UNKNOWN_TELESCOPE
from ctapipe.containers import MCHeaderContainer
from ctapipe.containers import *
from ctapipe.containers import IntensityStatisticsContainer, PeakTimeStatisticsContainer
from ctapipe.io.eventsource import EventSource
from ctapipe.io.datalevels import DataLevel
from astropy.coordinates import Angle
import astropy.units as u


class DL1EventSource(EventSource):
    def __init__(
        self,
        input_url,
        config=None,
        parent=None,
        **kwargs
    ):
        """
        EventSource for dl1 files in the standard DL1 data format

        Parameters:
        -----------
        input_url : str
            Path of the file to load
        config: ??
        parent: ??
        kwargs
        """
        super().init(
            input_url=input_url,
            config=config,
            parent=parent,
            **kwargs
        )

        self.file = tables.open_file(input_url)
        self._subarray_info = self._prepare_subarray_info()
        self.mc_header = self._parse_mc_header()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.file_.close()

    @staticmethod
    def is_compatible(file_path)
        """
        Implementation needed!
        """
        return False

    @property
    def is_simulation(self)
        """
        Implementation needed!
        """
        return True

    @property
    def subarray(self):
        return self._subarray_info

    @property
    def datalevels(self):
        return (DataLevel.DL1_IMAGES, DataLevel.DL1_PARAMETERS)  ## need to check if they are in file?

    @property
    def obs_id(self):
        return set(self.file_.root.dl1.event.subarray.trigger.col("obs_id"))

    @property
    def mc_header(self):
        return self._mc_header

    def _generator(self):
        try:
            yield from self._generate_events()
        except EOFError:
            msg = 'EOFError reading from "{input_url}". Might be truncated'.format(
                input_url=self.input_url
            )
            self.log.warning(msg)
            warnings.warn(msg)

    def prepare_subarray_info(self):
        """
        Implementation needed
        """
        return None

    def _parse_mc_header(self):
        """
        Implementation needed
        """
        return None

    def _generate_events(self):
        """
        Implementation needed
        """
        return None

    def _fill_trigger_info(self, data, array_event):
        """
        Implementation needed
        """
        return None

    def _fill_pointing(self, data):
        """
        Implementaion needed
        """
        return None
