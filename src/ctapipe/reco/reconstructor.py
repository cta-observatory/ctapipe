import pathlib
import weakref
from abc import abstractmethod
from enum import Flag, auto

import astropy.units as u
import joblib
import numpy as np
from astropy.coordinates import AltAz, SkyCoord

from ctapipe.containers import ArrayEventContainer, TelescopeImpactParameterContainer
from ctapipe.core import Component, Provenance, QualityQuery, TelescopeComponent
from ctapipe.core.traits import Integer, List, Path

from ..coordinates import shower_impact_distance

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
        return f"{self.name.lower()}"


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

    n_jobs = Integer(
        default_value=None,
        allow_none=True,
        help="Number of threads to use for the reconstruction if supported by the reconstructor.",
    ).tag(config=True)

    load_path = Path(
        default_value=None,
        allow_none=True,
        help="If given, load serialized model from this path.",
    ).tag(config=True)

    def __init__(self, subarray, atmosphere_profile=None, **kwargs):
        # Run the Component __init__ first to handle the configuration
        # and make `self.load_path` available
        Component.__init__(self, **kwargs)

        if self.load_path is None:
            self.subarray = subarray
            self.quality_query = StereoQualityQuery(parent=self)
            self.atmosphere_profile = atmosphere_profile
        else:
            loaded = self.read(self.load_path, subarray=subarray, **kwargs)
            self.__dict__.update(loaded.__dict__)

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

    def write(self, dictionary, path, overwrite=False):
        """
        Save a dictionary using joblib-pickle, which should contain all
        information/settings about an instance of a reconstructor (subclass).

        Parameters
        ----------
        dictionary : dict
            Dictionary to be saved. It can contain as many entries as needed,
            but must at least include the following:
            "name": Name of the ``Reconstructor`` subclass,
            "meta": Additional metadata
        path : str or pathlib.Path
            Path to which the dictionary will be saved.
        overwrite : Bool
            Whether to overwrite, if ``path`` already exists.
        """
        path = pathlib.Path(path)

        if path.exists() and not overwrite:
            raise OSError(f"Path {path} exists and overwrite=False")

        with path.open("wb") as f:
            Provenance().add_output_file(path, role="reconstructor")
            joblib.dump(dictionary, f, compress=True)

    @classmethod
    def read(cls, path, parent=None, subarray=None, **kwargs):
        """
        Read a dictionary from ``path`` containing all necessary information
        to construct an instance of a reconstructor (subclass).

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a dictionary containing all information about a
            ``Reconstructor`` (subclass).
        parent : None or Component or Tool
            Attach a new parent to the loaded class.
        subarray : SubarrayDescription
            Attach a new subarray to the loaded reconstructor
            A warning will be raised if the telescope types of the
            subarray stored in the pickled class do not match with the
            provided subarray.

        **kwargs are set on the constructed instance

        Returns
        -------
        Reconstructor instance
        """
        with open(path, "rb") as f:
            dictionary = joblib.load(f)

        meta = dictionary.pop("meta")
        name = dictionary.pop("name")
        loaded_subarray = dictionary.pop("subarray")
        instance = Reconstructor.from_name(name, subarray=loaded_subarray)

        for attr, value in dictionary.items():
            setattr(instance, attr, value)

        # first deal with kwargs that would need "special" treatment, parent and subarray
        if parent is not None:
            instance.parent = weakref.proxy(parent)
            instance.log = parent.log.getChild(name)

        if subarray is not None:
            if instance.subarray.telescope_types != subarray.telescope_types:
                instance.log.warning(
                    "Supplied subarray has different telescopes than subarray loaded from file"
                )
            instance.subarray = subarray

        for attr, value in kwargs.items():
            setattr(instance, attr, value)

        Provenance().add_input_file(path, role="reconstructor", reference_meta=meta)
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
                alt=event.pointing.tel[tel_id].altitude,
                az=event.pointing.tel[tel_id].azimuth,
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
            event.dl2.tel[tel_id].impact[
                self.__class__.__name__
            ] = TelescopeImpactParameterContainer(
                distance=impact_distances[tel_index],
                prefix=prefix,
            )
