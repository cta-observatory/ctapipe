from abc import abstractmethod

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, SkyCoord

from ctapipe.containers import ArrayEventContainer, TelescopeImpactParameterContainer
from ctapipe.core import QualityQuery, TelescopeComponent
from ctapipe.core.traits import List

from ..coordinates import shower_impact_distance

__all__ = [
    "Reconstructor",
    "GeometryReconstructor",
    "TooFewTelescopesException",
    "InvalidWidthException",
]


class TooFewTelescopesException(Exception):
    pass


class InvalidWidthException(Exception):
    pass


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

    #: ctapipe_rco entry points may provide Reconstructor implementations
    plugin_entry_point = "ctapipe_reco"

    def __init__(self, subarray, **kwargs):
        super().__init__(subarray=subarray, **kwargs)
        self.quality_query = StereoQualityQuery(parent=self, subarray=subarray)

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


class GeometryReconstructor(Reconstructor):
    """
    Base class for algorithms predicting only the shower geometry
    """

    def _create_hillas_dict(self, event):
        hillas_dict = {
            tel_id: dl1.parameters.hillas
            for tel_id, dl1 in event.dl1.tel.items()
            if all(self.quality_query(parameters=dl1.parameters, key=tel_id))
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
