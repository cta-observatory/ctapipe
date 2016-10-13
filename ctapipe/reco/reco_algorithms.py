from ctapipe.core import Component
from ctapipe.io.containers import RecoShowerGeom


class DirectionAlgorithm(Component):
    """This is the base class from which all direction reconstruction algorithms should inherit from"""

    def __init__(self, model = None):
        pass

    def predict(self,tels_dict):
        """overwrite this method with your favourite direction reconstruction algorithm

        Parameters:
        -----------
        tels_dict
        TODO:

        Returns:
        --------
        TODO:
        """
        return RecoShowerGeom()