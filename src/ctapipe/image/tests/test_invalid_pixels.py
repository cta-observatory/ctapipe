import numpy as np

from ctapipe.io.eventsource import EventSource


def test_neighbor_average(prod5_gamma_simtel_path):
    from ctapipe.image.invalid_pixels import NeighborAverage

    with EventSource(prod5_gamma_simtel_path) as source:
        subarray = source.subarray
        geometry = subarray.tel[1].camera.geometry

    neighbor_average = NeighborAverage(subarray)
    border = geometry.get_border_pixel_mask(1)

    ### 1 GAIN ###
    broken_pixels = np.zeros((1, geometry.n_pixels), dtype=bool)

    # one border pixel, one non-border pixel
    broken_pixels[:, np.nonzero(border)[0][0]] = True
    broken_pixels[:, np.nonzero(~border)[0][0]] = True

    image = np.ones((geometry.n_pixels))
    image[broken_pixels[0]] = 9999
    peak_time = np.full(geometry.n_pixels, 20.0)
    peak_time[broken_pixels[0]] = -1

    interpolated_image, interpolated_peaktime = neighbor_average(
        tel_id=1, image=image, peak_time=peak_time, pixel_mask=broken_pixels
    )

    assert np.allclose(interpolated_image[broken_pixels[0]], 1.0)
    assert np.allclose(interpolated_peaktime[broken_pixels[0]], 20.0)

    ### 2 GAIN ###
    broken_pixels = np.zeros((2, geometry.n_pixels), dtype=bool)

    # one border pixel, one non-border pixel
    broken_pixels[:, np.nonzero(border)[0][0]] = True
    broken_pixels[:, np.nonzero(~border)[0][0]] = True

    image = np.ones((2, geometry.n_pixels))
    image[broken_pixels] = 9999
    peak_time = np.full((2, geometry.n_pixels), 20.0)
    peak_time[broken_pixels] = -1

    interpolated_image, interpolated_peaktime = neighbor_average(
        tel_id=1, image=image, peak_time=peak_time, pixel_mask=broken_pixels
    )

    assert np.allclose(interpolated_image[broken_pixels], 1.0)
    assert np.allclose(interpolated_peaktime[broken_pixels], 20.0)


def test_negative_image(prod5_gamma_simtel_path):
    """Test with negative charges"""
    from ctapipe.image.invalid_pixels import NeighborAverage

    with EventSource(prod5_gamma_simtel_path) as source:
        subarray = source.subarray
        geometry = subarray.tel[1].camera.geometry

    neighbor_average = NeighborAverage(subarray)

    broken_pixels = np.zeros((1, geometry.n_pixels), dtype=bool)

    # one border pixel, one non-border pixel
    border = geometry.get_border_pixel_mask(1)
    broken_pixels[:, np.nonzero(border)[0][0]] = True
    broken_pixels[:, np.nonzero(~border)[0][0]] = True

    image = np.full(geometry.n_pixels, -10.0)
    image[broken_pixels[0]] = 9999
    peak_time = np.full(geometry.n_pixels, 20.0)
    peak_time[broken_pixels[0]] = -1

    interpolated_image, interpolated_peaktime = neighbor_average(
        tel_id=1, image=image, peak_time=peak_time, pixel_mask=broken_pixels
    )
    assert np.allclose(interpolated_image[broken_pixels[0]], -10)
    assert np.allclose(interpolated_peaktime[broken_pixels[0]], 20.0)
