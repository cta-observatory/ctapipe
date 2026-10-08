import warnings
import weakref
import zipfile
from abc import abstractmethod
from enum import Flag, auto

import astropy.units as u
import joblib
import numpy as np
from astropy.coordinates import AltAz, SkyCoord

from ..containers import ArrayEventContainer, TelescopeImpactParameterContainer
from ..coordinates import shower_impact_distance
from ..core import Provenance, QualityQuery, TelescopeComponent
from ..core.traits import Int, List
from ..io.models import ZipModelReader
from ..utils.deprecation import CTAPipeDeprecationWarning

__all__ = [
    "Reconstructor",
    "HillasGeometryReconstructor",
    "TooFewTelescopesException",
    "InvalidWidthException",
    "ReconstructionProperty",
]


class ReconstructionProperty(Flag):
    """
    Primary particle properties estimated by a `Reconstructor`

    These properties are of enum.Flag type and can thus be
    combined using bitwise operators to indicate a reconstructor
    provides several properties at once.
    """

    #: Energy if the primary particle
    ENERGY = auto()
    #: Geometric properties of the primary particle,
    #: direction and impact point
    GEOMETRY = auto()
    #: Prediction score that a particle belongs to a certain class
    PARTICLE_TYPE = auto()
    #: Disp, distance of the source position from the Hillas COG along the main axis
    DISP = auto()

    def __str__(self):
        return self.name.lower()


class TooFewTelescopesException(Exception):
    """
    Less valid telescope events than required in an array event.
    """


class InvalidWidthException(Exception):
    """Hillas width is 0 or nan"""


class StereoQualityQuery(QualityQuery):
    """Quality criteria for dl1 parameters checked for telescope events to enter
    into stereo reconstruction"""

    quality_criteria = List(
        default_value=[
            ("> 50 phe", "parameters.hillas.intensity > 50"),
            ("Positive width", "parameters.hillas.width.value > 0"),
            ("> 3 pixels", "parameters.morphology.n_pixels > 3"),
        ],
        help=QualityQuery.quality_criteria.help,
    ).tag(config=True)


class Reconstructor(TelescopeComponent):
    """
    This is the base class from which all reconstruction
    algorithms should inherit from
    """

    #: ctapipe_reco entry points may provide Reconstructor implementations
    plugin_entry_point = "ctapipe_reco"

    n_jobs = Int(
        default_value=None,
        allow_none=True,
        help="Number of threads to use for the reconstruction if supported by the reconstructor.",
    ).tag(config=True)

    def __init__(self, subarray, atmosphere_profile=None, **kwargs):
        super().__init__(subarray=subarray, **kwargs)
        self.quality_query = StereoQualityQuery(parent=self)
        self.atmosphere_profile = atmosphere_profile

    @property
    @abstractmethod
    def needs_atmosphere_profile(self) -> bool:
        """Whether this reconstructor requires ``atmosphere_profile``."""

    @abstractmethod
    def __call__(self, event: ArrayEventContainer):
        """
        Perform stereo reconstruction on event.

        This method must fill the result of the reconstruction into the
        dl2 structure of the event.

        Parameters
        ----------
        event : `ctapipe.containers.ArrayEventContainer`
            The event, needs to have dl1 parameters.
            Will be filled with the corresponding dl2 containers,
            reconstructed stereo geometry and telescope-wise impact position.
        """

    @classmethod
    def read(cls, path, parent=None, subarray=None, **kwargs):
        """Read a joblib-pickled reconstructor from ``path``

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a Reconstructor instance pickled using joblib
        parent : None or Component or Tool
            Attach a new parent to the loaded class, this will properly
        subarray : SubarrayDescription
            Attach a new subarray to the loaded reconstructor
            A warning will be raised if the telescope types of the
            subarray stored in the pickled class do not match with the
            provided subarray.

        **kwargs are set on the loaded instance

        Returns
        -------
        Reconstructor instance loaded from file
        """
        # new zip-based format
        if zipfile.is_zipfile(path):
            with ZipModelReader(path) as zip_reader:
                instance = zip_reader.read_reconstructor()
                instance.load_path = path
                meta = zip_reader.meta
            Provenance().add_input_file(path, role="reconstructor", reference_meta=meta)
        else:
            warnings.warn(
                "Storing reconstructors using pickle directly is deprecated, used ctapipe.io.ZipModelWriter instead.",
                CTAPipeDeprecationWarning,
            )
            instance = joblib.load(path)
            # no meta in joblib files
            Provenance().add_input_file(path, role="reconstructor", add_meta=False)

        if not isinstance(instance, cls):
            raise TypeError(
                f"{path} did not contain an instance of {cls}, got {instance}"
            )

        # first deal with kwargs that would need "special" treatmet, parent and subarray
        if parent is not None:
            instance.parent = weakref.proxy(parent)
            instance.log = parent.log.getChild(instance.__class__.__name__)

        if subarray is not None:
            if instance.subarray.telescope_types != subarray.telescope_types:
                instance.log.warning(
                    "Supplied subarray has different telescopes than subarray loaded from file"
                )
            instance.subarray = subarray

        for attr, value in kwargs.items():
            setattr(instance, attr, value)

        return instance


class HillasGeometryReconstructor(Reconstructor):
    """
    Base class for algorithms predicting only the shower geometry using Hillas Based methods
    """

    def _create_hillas_dict(self, event):
        hillas_dict = {
            tel_id: dl1.parameters.hillas
            for tel_id, dl1 in event.dl1.tel.items()
            if all(self.quality_query(parameters=dl1.parameters))
        }

        if len(hillas_dict) < 2:
            raise TooFewTelescopesException()

        # check for np.nan or 0 width's as these screw up weights
        if any([np.isnan(h.width.value) for h in hillas_dict.values()]):
            raise InvalidWidthException(
                "A HillasContainer contains an ellipse of width=np.nan"
            )

        if any([h.width.value == 0 for h in hillas_dict.values()]):
            raise InvalidWidthException(
                "A HillasContainer contains an ellipse of width=0"
            )

        return hillas_dict

    @staticmethod
    def _get_telescope_pointings(event):
        return {
            tel_id: SkyCoord(
                alt=event.monitoring.tel[tel_id].pointing.altitude,
                az=event.monitoring.tel[tel_id].pointing.azimuth,
                frame=AltAz(),
            )
            for tel_id in event.dl1.tel.keys()
        }

    def _store_impact_parameter(self, event):
        """Compute and store the impact parameter for each reconstruction."""
        geometry = event.dl2.stereo.geometry[self.__class__.__name__]

        if geometry.is_valid:
            impact_distances = shower_impact_distance(
                shower_geom=geometry,
                subarray=self.subarray,
            )
        else:
            n_tels = len(self.subarray)
            impact_distances = u.Quantity(np.full(n_tels, np.nan), u.m)

        default_prefix = TelescopeImpactParameterContainer.default_prefix
        prefix = f"{self.__class__.__name__}_tel_{default_prefix}"
        for tel_id in event.trigger.tels_with_trigger:
            tel_index = self.subarray.tel_indices[tel_id]
            event.dl2.tel[tel_id].impact[self.__class__.__name__] = (
                TelescopeImpactParameterContainer(
                    distance=impact_distances[tel_index],
                    prefix=prefix,
                )
            )
