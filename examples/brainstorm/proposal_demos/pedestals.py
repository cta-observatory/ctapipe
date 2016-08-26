"""
A module to demonstrate some simple calibration algorithms:
* the calculation of pedestals and pedestal variances
* the application of pedestal calibration to data
* the use of pedestal variances to identify broken pixels

It also implements a simple fake pedestal event generator
"""

import numpy as np


def make_fake_pedestal_images(nevents=100, npix=1024, ped_variance=50,
                              ped_offset=-1000,
                              noisy_pix=[10, 100, 740],
                              dead_pix=[6, 17, 45, 900]):
    """ generate fake raw data images """
    noise_pe = np.ones(shape=(npix), dtype=np.int64) * 30
    noise_pe[noisy_pix] = 80   # noisy pixels
    noise_pe[dead_pix] = 0     # dead pixels
    pedestal = np.random.normal(scale=ped_variance, size=npix) + ped_offset
    images = np.random.poisson(noise_pe, size=(nevents, npix)) + pedestal
    return images


def calc_pedestals(images_ped):
    """really simple pedestal calc. The input is a chunk of images that
    come from pedestal triggers (e.g. with no signal in them)"""
    peds = np.mean(images_ped, axis=0)
    pedvars = np.var(images_ped, axis=0)
    return (peds, pedvars)


def find_bad_pixels(pedvars, noisy_threshold=3.8, dead_threshold=-4.0):
    """returns list of noisy or dead pixels, based on distance in standard
    deviations of pedestal variance from the median value for all
    pixels
    """
    pedvar_std = np.std(pedvars)
    pedvars_ac = pedvars - np.median(pedvars)  # ac-coupled variances
    noisy_pix = pedvars_ac > (pedvar_std * noisy_threshold)
    dead_pix = pedvars_ac < (pedvar_std * dead_threshold)
    return noisy_pix, dead_pix


def apply_pedestals(peds, images):
    """ substract pedestals from all images """
    return images - peds


if __name__ == '__main__':

    import matplotlib.pyplot as plt  # plotting not needed until now

    # some example usage:

    images = make_fake_pedestal_images(nevents=200)
    pixids = np.arange(len(images[0]))

    peds, pedvars = calc_pedestals(images)
    images_pedsub = apply_pedestals(peds, images)
    noisy, dead = find_bad_pixels(pedvars)

    # =================================================================
    # Make some plots to visualize the results
    # =================================================================
    plt.style.use("ggplot")
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
    fig.suptitle("Measured from {} Pedestal-Trigger Events"
                 .format(images.shape[0]))

    # image values per pixel (for the first 3 pixels only)
    ax[0].set_ylabel("Raw images")
    for ii in range(3):
        ax[0].plot(pixids, images[ii], alpha=0.5, drawstyle="steps-mid",
                   label="Evt {:2d}".format(ii))
    ax[0].legend()

    # pedestal-subtracted image values per pixel (for the first 3
    # pixels only)
    ax[1].set_ylabel("Images-Ped")
    for ii in range(3):
        ax[1].plot(pixids, images_pedsub[ii],
                   alpha=0.5, drawstyle="steps-mid",
                   label="Evt {:2d}".format(ii))
    ax[1].legend()

    # pedestals
    ax[2].plot(peds, drawstyle="steps-mid")
    ax[2].set_ylabel("pedestal")

    # pedestal variances
    ax[3].plot(pixids, pedvars, drawstyle="steps-mid")
    ax[3].set_xlabel("pixel id")
    ax[3].set_ylabel("pedestal variance")

    # mark noisy and dead pixels that were detected:
    ax[3].scatter(pixids[noisy], pedvars[noisy], color='blue', label="noisy")
    ax[3].scatter(pixids[dead],  pedvars[dead], color='cyan', label="dead")
    ax[3].legend()

    fig2 = plt.figure()
    plt.hist(peds, bins=50, log=True)
    plt.title("Pedestals for All Pixels")
    plt.xlabel("Pedestal level")
    plt.ylabel("Frequency")

    plt.show()
