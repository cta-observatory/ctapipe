import onnxruntime

import numpy as np
from typing import List, Dict
from abc import abstractmethod

from ctapipe.reco.reco_algorithms import Reconstructor, ReconstructedShowerContainer

__all__ = ["ONNXModel", "DeepLearningReconstructor"]


class ONNXModel:
    def __init__(self, path):
        try:
            self.sess = onnxruntime.InferenceSession(path)
        except RuntimeError:
            raise ValueError(f'The model could not be loaded from "{path}"')

    def predict(self, *inputs, **named_inputs):
        """
        Start a prediction using the given inputs

        Parameters
        ----------
        inputs
            Ordered arguments to use as inputs
        named_inputs
            Named arguments to use as inputs
        Returns
        -------
        list
            List with the predictions of each output
        """
        n_inputs = len(self.inputs)
        if len(named_inputs) == 0 and len(inputs) == 0:
            raise ValueError(
                "The number of given (named xor ordered) arguments must be at least 1"
            )
        if len(named_inputs) != 0 and len(inputs) != 0:
            raise ValueError(
                "Ordered arguments and named arguments can't be given in the same prediction"
            )
        if len(named_inputs) != n_inputs and len(inputs) != n_inputs:
            raise ValueError(
                f"The number of given arguments ({max(len(named_inputs), len(inputs))}) must be "
                f"equal to the number of model inputs ({n_inputs})"
            )
        if len(inputs) != 0:
            for i in range(n_inputs):
                named_inputs[self.inputs[i].name] = inputs[i]

        n_predictions = [len(named_inputs[name]) for name in named_inputs]
        if len(set(n_predictions)) != 1:
            raise ValueError("All inputs must have the same length")

        for name in named_inputs:
            inp = named_inputs[name]
            if type(inp) is np.ndarray and inp.dtype == np.float64:
                named_inputs[name] = inp.astype(np.float32)
        return self.sess.run(None, named_inputs)

    @property
    def inputs(self):
        """
        Inputs as returned by the ONNX model
        """
        return self.sess.get_inputs()

    @property
    def outputs(self):
        """
        Outputs as returned by the ONNX model
        """
        return self.sess.get_outputs()


class DeepLearningReconstructor(Reconstructor):
    """
    Base class for techniques that use Deep Learning to reconstruct the direction and/or
    energy of an atmospheric shower using one or more ONNX models (neural networks).
    It transforms each event observation to the data used by the model and combines the
    outputs into a ReconstructedShowerContainer.

    Attributes
    ----------
    models_paths : Dict[str, str]
        dictionary with cam name as key and
        ONNX model file path as value
    """

    def __init__(self, models_paths, config=None, parent=None, **kwargs):
        self.models = dict(
            [(cam_name, ONNXModel(path)) for cam_name, path in models_paths.items()]
        )
        not_supported_cams = [
            cam_name
            for cam_name in self.models.keys()
            if cam_name not in self.supported_cameras
        ]
        if len(not_supported_cams) > 0:
            raise ValueError(
                f"Some of the given camera names are not supported by this reconstructor: "
                f"{', '.join(not_supported_cams)}"
            )
        super().__init__(config=config, parent=parent, **kwargs)

    @property
    @abstractmethod
    def supported_cameras(self):
        """
        Returns
        -------
        List[str]
            List of camera names supported by this reconstructor
        """

    @staticmethod
    def _get_tel_ids_with_cam(event, subarray, cam_name):
        """
        Get telescope IDs using a given camera

        Parameters
        ----------
        subarray : ctapipe.instrument.SubarrayDecription
            Subarray description
        cam_name : str
            Camera name for filter
        """
        for tel_id in event.dl1.tel:
            if subarray.tel[tel_id].camera.geometry.camera_name != cam_name:
                continue
            yield tel_id

    def predict(self, event, subarray, **kwargs):
        """
        Start a prediction using the model and then combine
        all the observations into a ReconstructedShowerContainer

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
            if models_path is None:
                continue

            model = self._get_model(cam_name)

            obs_inputs = [
                self._to_input(event, tel_id, cam_name, **kwargs)
                for tel_id in self._get_tel_ids_with_cam(event, subarray, cam_name)
            ]

            predictions[cam_name] = list()
            for inp in obs_inputs:
                if type(inp) is list:
                    predictions[cam_name].append(model.predict(*inp))
                elif type(inp) is dict:
                    predictions[cam_name].append(model.predict(**inp))
                else:
                    predictions[cam_name].append(model.predict(inp))
        return self._reconstruct(predictions)

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
    def _to_input(self, event, tel_id, cam_name, **kwargs):
        """
        Method to convert an observation of an event to the model input.

        Parameters
        ----------
        event : ctapipe.containers.EventAndMonDataContainer
            Event to be converted
        tel_id: int
            The telescope id of the observation to be converted.
        cam_name: str
            The camera name of the observation

        Returns
        -------
        Dict[str, str]
            Dictionary for the inputs, with the input name as the key, and the input array as the value
        """

    @abstractmethod
    def _reconstruct(self, models_outputs):
        """
        Method to convert the model outputs for one event to
        a container with the reconstructed shower.

        Parameters
        ----------
        models_outputs : Dict[str, List[Any]]
            dictionary with camera name as key and
            a list of model outputs as value

        Returns
        -------
        ReconstructedShowerContainer
            Reconstructed shower container made from the models' outputs
        """
