"""An example (and test) plugin for ctapipe providing an EventSource and a Reconstructor"""
import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation

from ctapipe.containers import (
    ObservationBlockContainer,
    ReconstructedGeometryContainer,
    SchedulingBlockContainer,
)
from ctapipe.core import traits
from ctapipe.instrument import (
    CameraDescription,
    CameraGeometry,
    CameraReadout,
    OpticsDescription,
    ReflectorShape,
    SizeType,
    SubarrayDescription,
    TelescopeDescription,
)
from ctapipe.io import DataLevel, EventSource
from ctapipe.io.datawriter import ArrayEventContainer
from ctapipe.reco import Reconstructor

__all__ = [
    "PluginEventSource",
    "PluginReconstructor",
]


optics = OpticsDescription(
    "plugin",
    size_type=SizeType.LST,
    n_mirrors=1,
    equivalent_focal_length=28 * u.m,
    effective_focal_length=29 * u.m,
    mirror_area=400 * u.m**2,
    n_mirror_tiles=250,
    reflector_shape=ReflectorShape.PARABOLIC,
)

step = 0.1
t = np.arange(0, 40, step)
geometry = CameraGeometry.make_rectangular()
camera = CameraDescription(
    name="plugin",
    geometry=geometry,
    readout=CameraReadout(
        name="plugin",
        sampling_rate=1 * u.GHz,
        reference_pulse_shape=np.exp(-0.5 * (t - 10) ** 2 / 4),
        reference_pulse_sample_width=step * u.ns,
        n_channels=2,
        n_pixels=geometry.n_pixels,
        n_samples=40,
    ),
)


telescope = TelescopeDescription("plugin", optics=optics, camera=camera)


subarray = SubarrayDescription(
    name="plugin",
    tel_descriptions={1: telescope},
    tel_positions={1: [0, 0, 0] * u.m},
    reference_location=EarthLocation(lat=0 * u.deg, lon=0 * u.deg, height=100 * u.m),
)


class PluginEventSource(EventSource):
    """A minimal plugin event source"""

    foo = traits.Bool(
        True, help="example traitlet to see that it is included in --help"
    ).tag(config=True)

    is_simulation = False
    datalevels = (DataLevel.DL1_IMAGES,)
    subarray = subarray
    observation_blocks = {1: ObservationBlockContainer(obs_id=1)}
    scheduling_blocks = {1: SchedulingBlockContainer(sb_id=1)}

    @classmethod
    def is_compatible(cls, path):
        """Foo"""
        return str(path).endswith(".plugin")

    def _generator(self):
        """Foo"""
        for i in range(10):
            yield ArrayEventContainer(count=i)


class PluginReconstructor(Reconstructor):
    """A plugin Reconstructor"""

    foo = traits.Bool(
        True, help="example traitlet to see that it is included in --help"
    ).tag(config=True)

    def __call__(self, event: ArrayEventContainer):
        """Foo"""
        event.dl2.geometry["PluginReconstructor"] = ReconstructedGeometryContainer()
