from typing import List, Dict
from abc import abstractmethod

from tensorflow import keras

from ctapipe.reco.reco_algorithms import Reconstructor, ReconstructedShowerContainer

__all__ = ['KerasReconstructor']


class KerasReconstructor(Reconstructor):
    """
    Class that reconstructs the direction and/or energy of an atmospheric shower
    using a Keras model (neural network) and a Sequence (generator) that transforms
    each event observation to the data used by the model and combines the outputs into a
    ReconstructedShowerContainer.

    Attributes
    ----------
    weights : Dict[str, str]
        dictionary with cam name as key and
        the H5 with the weights paths for the model as values
    """

    def __init__(self, models, config=None, parent=None, **kwargs):
        self.models = models
        if any(cam_name not in self.supported_cameras for cam_name in self.models.keys()):
            raise ValueError("Some of the weights telescope types are not supported by this reconstructor")
        super().__init__(config=config, parent=parent, **kwargs)

    @property
    @abstractmethod
    def supported_cameras(self):
        """
        Returns
        -------
        List[str]
            List of camera names available for this model
        """

    @staticmethod
    def _get_tel_ids_with_cam(subarray, cam_name):
        """
        Get telescope IDs using a given camera

        Parameters
        ----------
        subarray : ctapipe.instrument.SubarrayDecription
            Subarray description
        cam_name : str
            Camera name for filter
        """
        for tel_id in subarray.tel:
            if subarray.tel[tel_id].camera.geometry.camera_name != cam_name:
                continue
            yield tel_id

    def predict(self, event, subarray, **kwargs):
        """
        Method to start a prediction using the model and then combine
        all the observations into one

        Parameters
        ----------
        event : ctapipe.containers.EventAndMonDataContainer
            Event container from which the prediction will be made
        subarray : ctapipe.instrument.SubarrayDecription
            Subarray description
        """
        predictions = dict()
        for cam_name in self.supported_cameras:
            models_path = self.models.get(cam_name)
            if cam_name not in models_path:
                continue

            model = self._get_model(cam_name)

            inputs = [self._to_input(event, tel_id, **kwargs)
                      for tel_id in self._get_tel_ids_with_cam(subarray, cam_name)]

            predictions[cam_name] = list()
            for inp in inputs:
                predictions[cam_name].append(model.predict(inp))
        return self._combine(predictions)

    def _get_model(self, cam_name):
        """
        Return an instance of a Keras model with the architecture given a camera name.

        Parameters
        ----------
        cam_name : str
            Camera name (e.g. FlashCam, ASTRICam, LSTCam, etc.)

        Returns
        -------
        model : keras.Sequential
            List of events to be used for training
        """
        return self.models.get(cam_name)

    @abstractmethod
    def _to_input(self, event, tel_id, **kwargs):
        """
        Method to convert an observation of an event to the model input.

        Parameters
        ----------
        event : ctapipe.containers.EventAndMonDataContainer
            Event to be converted
        tel_id: int
            The telescope id of the observation to be converted. Will be None if predict_per_observation is False

        Returns
        -------
        list
            List of the model inputs
        """

    @abstractmethod
    def _combine(self, models_outputs):
        """
        Method to convert the model outputs for one event to
        a container with the reconstructed shower.

        Parameters
        ----------
        models_outputs : Dict[str, List[Any]]
            dictionary with tel type as key and
            an array of model outputs as value

        Returns
        -------
        ReconstructedShowerContainer
            Reconstructed shower container made from the model's output
            for one event
        """
