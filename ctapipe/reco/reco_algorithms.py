from abc import abstractmethod
import numpy as np

from ctapipe.core import Component, QualityQuery
from ctapipe.containers import ReconstructedGeometryContainer, ArrayEventContainer
from astropy.coordinates import SkyCoord, AltAz

from ctapipe.core.traits import List

__all__ = ["Reconstructor", "TooFewTelescopesException", "InvalidWidthException"]


class TooFewTelescopesException(Exception):
    pass


class InvalidWidthException(Exception):
    pass


class StereoQualityQuery(QualityQuery):
    """Quality criteria for dl1 parameters checked for telescope events to enter
    into stereo reconstruction"""

    quality_criteria = List(
        default_value=[
            ("> 50 phe", "lambda p: p.hillas.intensity > 50"),
            ("Positive width", "lambda p: p.hillas.width.value > 0"),
            ("> 3 pixels", "lambda p: p.morphology.num_pixels > 3"),
        ],
        help=QualityQuery.quality_criteria.help,
    ).tag(config=True)


class Reconstructor(Component):
    """
    This is the base class from which all direction reconstruction
    algorithms should inherit from
    """

    def __init__(self, subarray, **kwargs):
        super().__init__(**kwargs)
        self.subarray = subarray
        self.check_parameters = StereoQualityQuery(parent=self)

    @abstractmethod
    def __call__(self, event: ArrayEventContainer):
        """overwrite this method with your favourite direction reconstruction
        algorithm

        Parameters
        ----------
        tels_dict : dict
            general dictionary containing all triggered telescopes data

        Returns
        -------
        `~ctapipe.containers.ReconstructedGeometryContainer`

        """
        return ReconstructedGeometryContainer()

    def _create_hillas_dict(self, event):
        hillas_dict = {
            tel_id: dl1.parameters.hillas
            for tel_id, dl1 in event.dl1.tel.items()
            if all(self.check_parameters(dl1.parameters))
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
