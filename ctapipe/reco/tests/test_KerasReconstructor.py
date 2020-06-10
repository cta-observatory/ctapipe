import os
from typing import Union
from pandas import np
from tensorflow import keras

from ctapipe.containers import ReconstructedShowerContainer
from ctapipe.io import event_source
from ctapipe.utils import get_dataset_path
from ctapipe.calib import CameraCalibrator
from ctapipe.reco.KerasReconstructor import KerasReconstructor


def gen_double_input_nn(n_neurons, input_dim, output_dim, initializer='random_uniform'):
    """
    A simple generator of a Keras model with two inputs and one output.

    Parameters
    ----------
    n_neurons: int
        Number of neurons for the Dense layer
    input_dim: Union[tuple,int]
        Inputs shape
    output_dim: Union[tuple,int]
        Output shape
    initializer
        Keras initializer for the weights

    Returns
    -------
        keras.Model
            The generated Keras model
    """
    inp_a = keras.Input((input_dim,))
    layer_a = keras.layers.Dense(n_neurons, kernel_initializer=initializer, bias_initializer=initializer)
    x_a = layer_a(inp_a)

    inp_b = keras.Input((input_dim,))
    layer_b = keras.layers.Dense(n_neurons, kernel_initializer=initializer, bias_initializer=initializer)
    x_b = layer_b(inp_a)

    x = keras.layers.concatenate([x_a, x_b])
    out = keras.layers.Dense(output_dim)(x)
    return keras.Model(inputs=[inp_a, inp_b], outputs=out)


class TKerasReconstructor(KerasReconstructor):
    """
    A test Keras reconstructor
    """

    @property
    def supported_cameras(self):
        """
        This reconstructor will support observations made with FlashCam and ASTRICam

        Returns
        -------
        list
        """
        return ['FlashCam', 'ASTRICam']

    def _to_input(self, event, tel_id, **kwargs):
        """
        Each observation will have a random input

        Parameters
        ----------
        event
        tel_id
        kwargs

        Returns
        -------

        """
        return [[np.random.rand(3)], [np.random.rand(3)]]

    def _combine(self, models_outputs):
        """
        Before combining an assert will be made to test if results are as expected

        Parameters
        ----------
        models_outputs

        Returns
        -------

        """
        for cam_name in models_outputs:
            assert cam_name in self.supported_cameras
            predictions = models_outputs[cam_name]
            for pred in predictions:
                assert np.all(pred == 0.)
        return ReconstructedShowerContainer()


def test_filter_cams():
    """
    Tests the telescope filtering by camera name
    """
    filename = get_dataset_path("gamma_test_large.simtel.gz")

    source = event_source(filename, max_events=10)
    calib = CameraCalibrator(subarray=source.subarray)

    subarray = source.subarray
    for event in source:
        calib(event)
        count = 0
        for cam in subarray.camera_types:
            cam_name = cam.camera_name
            tel_ids = list(KerasReconstructor._get_tel_ids_with_cam(subarray=subarray, cam_name=cam_name))
            for tel_id in tel_ids:
                assert subarray.tel[tel_id].camera.geometry.camera_name == cam_name
            count += len(tel_ids)
        assert count == len(subarray.tel)


def test_reconstruction():
    """
    Tests reconstruction is working using FlashCam and ASTRICam.
    """

    # save weights that should return 0 in h5
    models_path = 'models'
    try:
        os.makedirs(models_path)
    except OSError:
        pass
    flashcam_model = gen_double_input_nn(10, 3, 4)
    flashcam_path = os.path.join(models_path, 'flashcam.h5')
    flashcam_model.save(flashcam_path)

    astricam_model = gen_double_input_nn(20, 3, 4)
    astricam_path = os.path.join(models_path, 'astricam.h5')
    astricam_model.save(astricam_path)

    # load dataset
    filename = get_dataset_path("gamma_test_large.simtel.gz")

    source = event_source(filename, max_events=10)
    calib = CameraCalibrator(subarray=source.subarray)
    fit = TKerasReconstructor({
        'FlashCam': flashcam_path,
        'ASTRICam': astricam_path
    })

    # start prediction per event
    for event in source:
        calib(event)
        prediction = fit.predict(event, source.subarray)
        assert isinstance(prediction, ReconstructedShowerContainer)
