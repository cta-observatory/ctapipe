from numpy import isfinite

from ctapipe.calib import CameraCalibrator
from ctapipe.image.image_processor import ImageProcessor
from ctapipe.reco import HillasReconstructor
from ctapipe.reco.shower_reconstructor import Model3DGeometryReconstructor


def test_reconstruction(subarray_and_event_gamma_off_axis_500_gev):
    subarray, event = subarray_and_event_gamma_off_axis_500_gev

    calib = CameraCalibrator(subarray)
    image_processor = ImageProcessor(subarray)
    reconstructor = HillasReconstructor(subarray)
    model3d_reconstructor = Model3DGeometryReconstructor(subarray)

    calib(event)
    image_processor(event)
    reconstructor(event)
    model3d_reconstructor(event)

    assert model3d_reconstructor.__class__.__name__ in event.dl2.stereo.geometry
    geometryContainer = event.dl2.stereo.geometry[
        model3d_reconstructor.__class__.__name__
    ]
    assert isfinite(geometryContainer.total_photons)
    assert isfinite(geometryContainer.total_photons_uncert)
    assert isfinite(geometryContainer.core_x)
    assert isfinite(geometryContainer.core_uncert_x)
    assert isfinite(geometryContainer.core_y)
    assert isfinite(geometryContainer.core_uncert_y)
    assert isfinite(geometryContainer.alt)
    assert isfinite(geometryContainer.alt_uncert)
    assert isfinite(geometryContainer.az)
    assert isfinite(geometryContainer.az_uncert)
    assert isfinite(geometryContainer.h_max)
    assert isfinite(geometryContainer.h_max_uncert)
    assert isfinite(geometryContainer.width)
    assert isfinite(geometryContainer.width_uncert)
    assert isfinite(geometryContainer.length)
    assert isfinite(geometryContainer.length_uncert)
