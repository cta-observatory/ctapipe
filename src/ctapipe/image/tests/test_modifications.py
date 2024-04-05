import numpy as np

from ctapipe.image import modifications
from ctapipe.instrument import CameraGeometry


def test_add_noise():
    """
    Test that adding noise changes a dummy
    image in the expected way.
    """
    image = np.array([0, 0, 5, 1, 0, 0])
    rng = np.random.default_rng(42)
    # test different noise per pixel:
    noise = [6, 8, 0, 7, 9, 12]
    noisy = modifications._add_noise(image, noise, rng, correct_bias=False)
    assert image[2] == noisy[2]
    # For other seeds there exists a probability > 0 for no noise added at all
    assert noisy.sum() > image.sum()

    # test scalar
    noisy = modifications._add_noise(image, 20, rng, correct_bias=False)
    diff_no_bias = noisy - image
    assert (noisy > image).all()

    # test bias
    noisy = modifications._add_noise(image, 20, rng, correct_bias=True)
    assert np.sum(diff_no_bias) > np.sum(noisy - image)


def test_smear_image(prod5_lst):
    """
    Test that smearing the image leads to the expected results.
    For random smearing this will not work with all seeds.
    With the selected seed it will not lose charge
    (No pixel outside of the camera receives light) but
    for positive smear factors the image will be different
    from the input
    (At least for one pixel a positive charge is selected to
    be distributed among neighbors).
    """
    seed = 20

    # Hexagonal geometry -> That's why we divide by 6 below
    geom: CameraGeometry = prod5_lst.camera.geometry
    image = np.zeros_like(geom.pix_id, dtype=np.float64)
    # select two pixels, one at the edge with only 5 neighbors

    border_mask = geom.get_border_pixel_mask()
    # one border pixel, one central pixel
    signal_pixels = [np.nonzero(~border_mask)[0][0], np.nonzero(border_mask)[0][0]]

    signal_value = 10

    for signal_pixel in signal_pixels:
        for signal_value in [1, 10]:
            image[signal_pixel] = signal_value
            for fraction in [0, 0.2, 1]:
                smeared = modifications._smear_psf_randomly(
                    image,
                    fraction=fraction,
                    indices=geom.neighbor_matrix_sparse.indices,
                    indptr=geom.neighbor_matrix_sparse.indptr,
                    smear_probabilities=np.full(6, 1 / 6),
                    seed=seed,
                )

                # we may loose charge at the border of the camera
                if border_mask[signal_pixel]:
                    assert image.sum() >= smeared.sum()
                else:
                    assert image.sum() == smeared.sum()

                if fraction > 0:
                    assert image[signal_pixel] >= smeared[signal_pixel]


def test_defaults_no_change(example_subarray):
    """Test that the default settings do not change the input image"""
    rng = np.random.default_rng(0)

    modifier = modifications.ImageModifier(example_subarray)
    tel_id = 1
    n_pixels = example_subarray.tel[tel_id].camera.geometry.n_pixels
    image = rng.normal(50, 15, size=n_pixels).astype(np.float32)

    new_image = modifier(tel_id=tel_id, image=image)
    assert np.all(image == new_image)
