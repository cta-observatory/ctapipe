import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from tqdm.auto import tqdm

from ctapipe.image import hillas_parameters
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.toymodel import Gaussian
from ctapipe.instrument import CameraGeometry

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")

rng = np.random.default_rng(0)

# cam = CameraGeometry.from_name("LSTCam")
cam = CameraGeometry.from_name("NectarCam")
# cam = CameraGeometry.from_name("SCTCam")

true_width = 0.05 * u.m
true_length = 0.3 * u.m
true_psi = 45 * u.deg
true_x = 0.5 * u.m
true_y = -0.2 * u.m

image_intensity = 1000
test_nsb_level_pe = 3
# test_nsb_level_pe = 5

# n_sample = 10
n_sample = 1000

model = Gaussian(true_x, true_y, true_length, true_width, true_psi)


def sample_no_noise_no_cleaning():
    _, signal, _ = model.generate_image(
        cam, intensity=image_intensity, nsb_level_pe=0, rng=rng
    )
    h = hillas_parameters(cam, signal)
    return h


def sample_no_noise_with_cleaning():
    _, signal, _ = model.generate_image(
        cam, intensity=image_intensity, nsb_level_pe=0, rng=rng
    )

    mask = tailcuts_clean(
        cam,
        signal,
        3.0 * test_nsb_level_pe,
        2.0 * test_nsb_level_pe,
        min_number_picture_neighbors=2,
    )

    image_clean = np.zeros_like(signal)
    for pix in range(0, len(signal)):
        image_clean[pix] = max(0.0, signal[pix] - 2.0 * test_nsb_level_pe)

    h = hillas_parameters(cam[mask], image_clean[mask])
    return h


def sample_noise_with_cleaning():
    image, _, _ = model.generate_image(
        cam, intensity=image_intensity, nsb_level_pe=test_nsb_level_pe, rng=rng
    )

    mask = tailcuts_clean(
        cam,
        image,
        3.0 * test_nsb_level_pe,
        2.0 * test_nsb_level_pe,
        min_number_picture_neighbors=2,
    )

    image_clean = np.zeros_like(image)
    for pix in range(0, len(image)):
        image_clean[pix] = max(0.0, image[pix] - 2.0 * test_nsb_level_pe)

    h = hillas_parameters(cam[mask], image_clean[mask])
    return h


trials_no_noise_no_cleaning = [
    sample_no_noise_no_cleaning() for _ in tqdm(range(n_sample))
]
trials_no_noise_with_cleaning = [
    sample_no_noise_with_cleaning() for _ in tqdm(range(n_sample))
]
trials_noise_cleaning = [sample_noise_with_cleaning() for _ in tqdm(range(n_sample))]

titles = [
    "No Noise, all Pixels",
    f"No Noise, Tailcuts({test_nsb_level_pe*3}, {test_nsb_level_pe*2})",
    f"With Noise ({test_nsb_level_pe} p.e.), Tailcuts({test_nsb_level_pe*3}, {test_nsb_level_pe*2})",
]
values = [
    trials_no_noise_no_cleaning,
    trials_no_noise_with_cleaning,
    trials_noise_cleaning,
]

fig, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
for ax, trials, title in zip(axs, values, titles):
    pred = np.array([t.psi.to_value(u.rad) for t in trials])
    unc = np.array([t.psi_uncertainty.to_value(u.rad) for t in trials])
    limits = np.quantile(pred, [0.001, 0.999])
    hist, edges, plot = ax.hist(pred, bins=51, range=limits, density=True)
    x = np.linspace(edges[0], edges[-1], 500)
    ax.plot(x, norm.pdf(x, pred.mean(), pred.std()))
    ax.plot(x, norm.pdf(x, pred.mean(), unc.mean()))
    ax.set_title(title)
axs[2].set_xlabel("Psi / rad")
fig.savefig(f"{ctapipe_output}/output_plots/hillas_uncertainties.png", dpi=300)
