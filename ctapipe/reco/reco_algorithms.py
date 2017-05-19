from ctapipe.core import Component
from ctapipe.io.containers import ReconstructedShowerContainer, \
    ReconstructedEnergyContainer

__all__ = ['Reconstructor']


class Reconstructor(Component):
    """This is the base class from which all direction reconstruction
algorithms should inherit from"""

    def predict(self, tels_dict):
        """overwrite this method with your favourite direction reconstruction
        algorithm

        Parameters
        ----------
        tels_dict : dict
            general dictionary containing all triggered telescopes data

        Returns
        -------
        Standard  `RecoShowerGeom` container

        """
        return ReconstructedShowerContainer()


class EnergyReconstructor(Component):
    """This is the base class from which all energy reconstruction
algorithms should inherit from"""

    def predict(self, tels_dict):
        """overwrite this method with your favourite direction reconstruction
        algorithm

        Parameters
        ----------
        tels_dict : dict
            general dictionary containing all triggered telescopes data

        Returns
        -------
        Standard  `ReconstructedEnergyContainer` container

        """
        return ReconstructedEnergyContainer()


