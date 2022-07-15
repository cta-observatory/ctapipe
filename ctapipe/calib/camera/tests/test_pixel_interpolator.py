import numpy as np

from ctapipe.io.eventsource import EventSource


def test_interpolate_pixels(prod5_gamma_simtel_path):
    from ctapipe.calib.camera.pixel_interpolator import interpolate_pixels

    with EventSource(prod5_gamma_simtel_path) as source:
        subarray = source.subarray
        geometry = subarray.tel[1].camera.geometry

    broken_pixels = np.zeros(geometry.n_pixels, dtype=bool)

    # one border pixel, one non-border pixel
    border = geometry.get_border_pixel_mask(1)
    broken_pixels[np.nonzero(border)[0][0]] = True
    broken_pixels[np.nonzero(~border)[0][0]] = True

    image = np.ones(geometry.n_pixels)
    image[broken_pixels] = 9999
    peak_time = np.full(geometry.n_pixels, 20.0)
    peak_time[broken_pixels] = -1

    interpolated_image, interpolated_peaktime = interpolate_pixels(
        image, peak_time, broken_pixels, geometry
    )

    assert np.allclose(interpolated_image[broken_pixels], 1.0)
    assert np.allclose(interpolated_peaktime[broken_pixels], 20.0)


def test_negative_image(prod5_gamma_simtel_path):
    """Test with negative charges"""
    from ctapipe.calib.camera.pixel_interpolator import interpolate_pixels

    with EventSource(prod5_gamma_simtel_path) as source:
        subarray = source.subarray
        geometry = subarray.tel[1].camera.geometry

    broken_pixels = np.zeros(geometry.n_pixels, dtype=bool)

    # one border pixel, one non-border pixel
    border = geometry.get_border_pixel_mask(1)
    broken_pixels[np.nonzero(border)[0][0]] = True
    broken_pixels[np.nonzero(~border)[0][0]] = True

    image = np.full(geometry.n_pixels, -10.0)
    image[broken_pixels] = 9999
    peak_time = np.full(geometry.n_pixels, 20.0)
    peak_time[broken_pixels] = -1

    interpolated_image, interpolated_peaktime = interpolate_pixels(
        image, peak_time, broken_pixels, geometry
    )
    assert np.allclose(interpolated_image[broken_pixels], -10)
    assert np.allclose(interpolated_peaktime[broken_pixels], 20.0)
