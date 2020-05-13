import os
from pandas import np
from tensorflow import keras

from ctapipe.containers import ReconstructedShowerContainer
from ctapipe.io import event_source
from ctapipe.utils import get_dataset_path
from ctapipe.calib import CameraCalibrator
from ctapipe.reco.KerasReconstructor import KerasReconstructor


def gen_double_input_nn(n_neurons, input_dim, output_dim, initializer='random_uniform'):
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

    @property
    def supported_cameras(self):
        return ['FlashCam', 'ASTRICam']

    def _get_model(self, cam_name):
        if cam_name == 'FlashCam':
            n_neurons = 10
        elif cam_name == 'ASTRICam':
            n_neurons = 20
        elif cam_name == 'LSTCam':
            n_neurons = 30
        else:
            raise ValueError("Invalid camera name")
        return gen_double_input_nn(n_neurons, 3, 4)

    def _to_input(self, event, tel_id, **kwargs):
        return [[np.random.rand(3)], [np.random.rand(3)]]

    def _combine(self, models_outputs):
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
    weights_path = 'weights'
    try:
        os.makedirs(weights_path)
    except OSError:
        pass
    model_10 = gen_double_input_nn(10, 3, 4, 'zeros')
    w10_path = os.path.join(weights_path, 'w10.h5')
    model_10.save_weights(w10_path)

    model_20 = gen_double_input_nn(20, 3, 4, 'zeros')
    w20_path = os.path.join(weights_path, 'w20.h5')
    model_20.save_weights(w20_path)

    # load dataset
    filename = get_dataset_path("gamma_test_large.simtel.gz")

    source = event_source(filename, max_events=10)
    calib = CameraCalibrator(subarray=source.subarray)
    fit = TKerasReconstructor({
        'FlashCam': w10_path,
        'ASTRICam': w20_path
    })

    # start prediction per event
    for event in source:
        calib(event)
        prediction = fit.predict(event, source.subarray)
        assert isinstance(prediction, ReconstructedShowerContainer)
