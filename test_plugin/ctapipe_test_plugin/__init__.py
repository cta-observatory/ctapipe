import astropy.units as u
import numpy as np

from ctapipe.instrument import (
    CameraDescription,
    CameraGeometry,
    CameraReadout,
    OpticsDescription,
    SubarrayDescription,
    TelescopeDescription,
)
from ctapipe.io import DataLevel, EventSource
from ctapipe.io.datawriter import ArrayEventContainer

optics = OpticsDescription(
    "dummy",
    num_mirrors=1,
    equivalent_focal_length=28 * u.m,
    effective_focal_length=29 * u.m,
    mirror_area=400 * u.m**2,
    num_mirror_tiles=250,
)

step = 0.1
t = np.arange(0, 40, step)
camera = CameraDescription(
    camera_name="dummy",
    geometry=CameraGeometry.make_rectangular(),
    readout=CameraReadout(
        camera_name="dummy",
        sampling_rate=1 * u.GHz,
        reference_pulse_shape=np.exp(-0.5 * (t - 10) ** 2 / 4),
        reference_pulse_sample_width=step * u.ns,
    ),
)


telescope = TelescopeDescription(
    "dummy",
    "LST",
    optics=optics,
    camera=camera,
)


subarray = SubarrayDescription(
    name="Dummy",
    tel_descriptions={1: telescope},
    tel_positions={1: [0, 0, 0] * u.m},
)


class PluginEventSource(EventSource):
    """A minimal dummy event source"""

    is_simulation = False
    datalevels = (DataLevel.DL1_IMAGES,)
    obs_ids = (1,)
    subarray = subarray

    @classmethod
    def is_compatible(cls, path):
        return str(path).endswith(".dummy")

    def _generator(self):
        for i in range(10):
            yield ArrayEventContainer(count=i)
