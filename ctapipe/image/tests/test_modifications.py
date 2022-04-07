import numpy as np
from ctapipe.instrument import CameraGeometry
from ctapipe.image import modifications


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


def test_smear_image():
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

    # Hexagonal geometry -> Thats why we divide by 6 below
    geom = CameraGeometry.from_name("LSTCam")
    image = np.zeros_like(geom.pix_id, dtype=np.float64)
    # select two pixels, one at the edge with only 5 neighbors
    signal_pixels = [1, 1853]
    neighbors = geom.neighbor_matrix[signal_pixels]

    for signal_value in [1, 5]:
        image[signal_pixels] = signal_value
        for fraction in [0, 0.2, 1]:
            # random smearing
            # The seed is important here (See below)
            smeared = modifications._smear_psf_randomly(
                image,
                fraction=fraction,
                indices=geom.neighbor_matrix_sparse.indices,
                indptr=geom.neighbor_matrix_sparse.indptr,
                smear_probabilities=np.full(6, 1 / 6),
                seed=seed,
            )
            neighbors_1 = smeared[neighbors[0]]

            # this can be False if the "pseudo neighbor" of pixel
            # 1853 is selected (outside of the camera)
            assert np.isclose(image.sum(), smeared.sum())
            assert np.isclose(np.sum(neighbors_1) + smeared[1], image[1])
            # this can be False if for both pixels a 0 is
            # drawn from the poissonian (especially with signal value 1)
            if fraction > 0:
                assert not ((image > 0) == (smeared > 0)).all()
