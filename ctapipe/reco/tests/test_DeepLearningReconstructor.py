import os

import pytest
from pandas import np

from ctapipe.containers import ReconstructedShowerContainer
from ctapipe.io import event_source
from ctapipe.utils import get_dataset_path
from ctapipe.calib import CameraCalibrator
from ctapipe.reco.DeepLearningReconstructor import DeepLearningReconstructor, ONNXModel

MODELS_PATH = os.path.join(os.path.dirname(__file__), "models")


class TReconstructor(DeepLearningReconstructor):
    """
    A test child class of Deep Learning reconstructor
    """

    @property
    def supported_cameras(self):
        """
        This reconstructor will support observations made with FlashCam and ASTRICam

        Returns
        -------
        list
            List of supported cameras: FlashCam and ASTRICam
        """
        return ["FlashCam", "ASTRICam"]

    def _to_input(self, event, tel_id, cam_name):
        """
        Each observation will use the dl1 image and a random "extra" input
        """
        return {"img": [event.dl1.tel[tel_id].image], "extra": [np.random.rand(10)]}

    def _reconstruct(self, models_outputs):
        """
        Before combining an assert will be made to test if results are as
        expected (all ones)
        """
        for cam_name in models_outputs:
            assert cam_name in self.supported_cameras
            predictions = models_outputs[cam_name]
            for pred in predictions:
                assert np.allclose(pred[0], np.ones((1, 1)))
                assert np.allclose(pred[1], np.ones((1, 1)))
        return ReconstructedShowerContainer()


def test_onnx_model_validations():
    """
    Test different input cases for an ONNX model that should raise an error:
        - No inputs given
        - Named and ordered arguments are given in the same prediction
        - The number of inputs given does not match with the number model inputs
        - Some inputs have different length
    """
    with pytest.raises(ValueError):
        ONNXModel(os.path.join(MODELS_PATH, "model.onnx"))

    model = ONNXModel(os.path.join(MODELS_PATH, "LSTCam.onnx"))

    input_img = np.random.rand(1, 1855)
    input_extra = np.random.rand(1, 10)

    with pytest.raises(
        ValueError,
        match=r"^The number of given \(named xor ordered\) arguments must be at least 1$",
    ):
        model.predict()

    with pytest.raises(
        ValueError,
        match=r"^Ordered arguments and named arguments can't be given in the same "
        r"prediction$",
    ):
        model.predict(input_img, extra=input_extra)

    with pytest.raises(
        ValueError,
        match=r"^The number of given arguments \(1\) must be equal to the number of model "
        r"inputs \(2\)$",
    ):
        model.predict(input_img)

    with pytest.raises(
        ValueError,
        match=r"^The number of given arguments \(1\) must be equal to the number of model "
        r"inputs \(2\)$",
    ):
        model.predict(img=input_img)

    with pytest.raises(ValueError, match="All inputs must have the same length$"):
        two_input_imgs = np.random.rand(2, 1855)
        model.predict(img=two_input_imgs, extra=input_extra)


def test_onnx_prediction():
    model = ONNXModel(os.path.join(MODELS_PATH, "LSTCam.onnx"))

    input_img = np.random.rand(1, 1855)
    input_extra = np.random.rand(1, 10)

    out_a, out_b = model.predict(img=input_img, extra=input_extra)
    assert np.allclose(out_a, np.ones((1, 1)))
    assert np.allclose(out_b, np.ones((1, 1)))

    out_a, out_b = model.predict(input_img, input_extra)
    assert np.allclose(out_a, np.ones((1, 1)))
    assert np.allclose(out_b, np.ones((1, 1)))


def test_filter_cams():
    """
    Tests the telescope filtering by camera name.

    Iterates over all events from the test file filtering by camera name
    using the `ONNXReconstructor._get_tel_ids_with_cam(subarray, cam_name)`
    method to assert that no event is wrongly filtered.
    """
    filename = get_dataset_path("gamma_test_large.simtel.gz")

    source = event_source(filename, max_events=10)
    calib = CameraCalibrator(subarray=source.subarray)

    subarray = source.subarray
    for event in source:
        calib(event)
        # count to assert that observations from all the telescopes have been processed
        count = 0
        for cam in subarray.camera_types:
            cam_name = cam.camera_name
            tel_ids = list(
                TReconstructor._get_tel_ids_with_cam(
                    event=event, subarray=subarray, cam_name=cam_name
                )
            )
            for tel_id in tel_ids:
                assert subarray.tel[tel_id].camera.geometry.camera_name == cam_name
            count += len(tel_ids)
        assert count == len(event.dl1.tel)


def test_reconstructor_validations():
    """
    Tests reconstructor models validation
    """
    flashcam_path = os.path.join(MODELS_PATH, "FlashCam.onnx")
    astricam_path = os.path.join(MODELS_PATH, "ASTRICam.onnx")
    lstcam_path = os.path.join(MODELS_PATH, "LSTCam.onnx")

    with pytest.raises(
        ValueError,
        match=r"^Some of the given camera names are not supported by this "
        r"reconstructor: LSTCam$",
    ):
        TReconstructor(
            {
                "FlashCam": flashcam_path,
                "ASTRICam": astricam_path,
                "LSTCam": lstcam_path,
            }
        )


def test_reconstruction():
    """
    Tests reconstruction is working using FlashCam and ASTRICam.

    Starts a prediction using the TONNXReconstructor, an ONNXReconstructor
    implementation
    """

    # load dataset
    filename = get_dataset_path("gamma_test_large.simtel.gz")
    source = event_source(filename, max_events=10)
    calib = CameraCalibrator(subarray=source.subarray)

    # instantiate reconstructor
    flashcam_path = os.path.join(MODELS_PATH, "FlashCam.onnx")
    astricam_path = os.path.join(MODELS_PATH, "ASTRICam.onnx")
    fit = TReconstructor({"FlashCam": flashcam_path, "ASTRICam": astricam_path})

    # start prediction per event
    for event in source:
        calib(event)
        prediction = fit.predict(event, source.subarray)
        assert isinstance(prediction, ReconstructedShowerContainer)
