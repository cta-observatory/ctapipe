from ctapipe.calib import CameraCalibrator
from ctapipe.image.image_processor import ImageProcessor
from ctapipe.reco import HillasReconstructor
from ctapipe.reco.shower_reconstructor import ShowerReconstructor


def test_reconstruction(subarray_and_event_gamma_off_axis_500_gev):
    subarray, event = subarray_and_event_gamma_off_axis_500_gev

    calib = CameraCalibrator(subarray)
    image_processor = ImageProcessor(subarray)
    reconstructor = HillasReconstructor(subarray)
    shower_reconstructor = ShowerReconstructor(subarray)

    calib(event)
    image_processor(event)
    reconstructor(event)
    shower_reconstructor(event)

    assert shower_reconstructor.__class__.__name__ in event.dl2.stereo.geometry
