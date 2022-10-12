from ctapipe.calib import CameraCalibrator
from ctapipe.image.image_processor import ImageProcessor
from ctapipe.reco import HillasReconstructor
from ctapipe.reco.shower_reconstructor import Model3DGeometryReconstuctor


def test_reconstruction(subarray_and_event_gamma_off_axis_500_gev):
    subarray, event = subarray_and_event_gamma_off_axis_500_gev

    calib = CameraCalibrator(subarray)
    image_processor = ImageProcessor(subarray)
    reconstructor = HillasReconstructor(subarray)
    model3d_reconstructor = Model3DGeometryReconstuctor(subarray)

    calib(event)
    image_processor(event)
    reconstructor(event)
    model3d_reconstructor(event)

    assert model3d_reconstructor.__class__.__name__ in event.dl2.stereo.geometry
