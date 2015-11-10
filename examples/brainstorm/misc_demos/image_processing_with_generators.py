import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import time
import matplotlib.animation as animation
from astropy import convolution as conv

TIMESTEP = 0.1


def gen_static_model(nmax=None):

    count = 0
    X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    dist1 = np.sqrt(X ** 2 + Y ** 2)
    dist2 = np.sqrt((X - 2.0) ** 2 + Y ** 2)
    model = stats.norm.pdf(dist1, 0.0, 2.0)
    model += 0.05 * stats.norm.pdf(dist2, 0.0, 0.1)
    while True:
        if nmax is not None and count >= nmax:
            break
        yield model
        count += 1


def gen_timevar_model(nmax=None):

    count = 0
    tt = 0
    X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    dist1 = np.sqrt(X ** 2 + Y ** 2)
    background = stats.norm.pdf(dist1, 0.0, 2.0)

    while True:
        if nmax is not None and count >= nmax:
            break

        posX = 2.0 + X + np.cos(0.1 * tt)
        posY = Y + np.sin(0.1 * tt)
        source = np.sin(tt / 2) ** 2 * stats.norm.pdf(np.sqrt(posX ** 2 + posY ** 2),
                                                      0.0, 0.1)
        yield background + source

        count += 1
        tt += TIMESTEP


def gen_background_model():
    X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    dist1 = np.sqrt(X ** 2 + Y ** 2)
    background = stats.norm.pdf(dist1, 0.0, 2.0)
    while True:
        yield background


def sample_from_model(model_stream):
    for model in model_stream:
        sampled = np.random.poisson(model)
        yield sampled


def sum_image(image_stream):
    # first time, just yield the image and initialize the sum
    imsum = next(image_stream).copy()
    yield imsum

    # subsequent times, yield the cumulative sum
    for image in image_stream:
        imsum += image
        yield imsum


def smooth_image(image_stream, stddev=2.0):
    kernel = conv.Gaussian2DKernel(stddev)
    for image in image_stream:
        yield conv.convolve(image, kernel)


def get_rad_profile(img):
    X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    X = X[:, 50:]
    Y = Y[:, 50:]
    subimg = img[:, 50:]
    dist = np.hypot(X, Y)
    return np.histogram((dist ** 2).flatten(), weights=subimg.flatten(),
                        range=[0, 7], bins=15)


def subtract_background(image_stream, background_stream):
    while True:
        try:
            im = next(image_stream)
            bg = next(background_stream)
            norm = im.sum() / bg.sum()
            yield (im - bg * norm)
        except:
            break


def sink(image_stream):
    counter = 0
    for image in image_stream:
        counter += 1
        if counter % 1000 == 0:
            print("processed ", counter, "images")


def lightcurve_binner(image_stream, dt=10):
    """ input: image, output: image with no modification
    returns the lightcurve, when the stream ends
    """

    pos = 0
    x = arange(dt)
    lc = np.zeros(dt)

    for image in image_stream:
        totflux = image.sum()
        lc[pos] = totflux
        pos += 1

        if pos >= dt:
            yield lc[lc > 0].mean()
            pos = 0


def calc_hillas(image_stream):
    """ inputs images, outputs parameters as dict """
    output = dict()
    output['TYPE'] = 'hillasparams'

    for image in image_stream:
        X, Y = np.meshgrid(np.linspace(-5, 5, image.shape[0]),
                           np.linspace(-5, 5, image.shape[1]))

    yield output


def lightcurve_sink(value_stream):
    overall_lc = list()
    for val in value_stream:
        overall_lc.append(val)

    return np.arange(len(overall_lc)), np.array(overall_lc)


def display_image_sink(image_stream, vmax=None, fig=None, interval=30):

    def update(iframe, axim, image_stream):
        im = next(image_stream)
        axim.set_array(im)
        if vmax is None:
            axim.set_clim(0, im.max() * 0.90)
        return axim,

    if fig is None:
        fig = plt.figure()

    image = next(image_stream)
    axim = plt.imshow(image, interpolation='nearest', vmax=vmax)
    ani = animation.FuncAnimation(fig, update, fargs=[axim, image_stream],
                                  blit=True, interval=interval)

    return ani


if __name__ == '__main__':

    plt.spectral()

    #mod = gen_static_model()
    mod = gen_timevar_model()

    img = sample_from_model(mod)
    cum = sum_image(img)

    cumsub = subtract_background(cum, gen_background_model())

    fig = plt.figure(figsize=(14, 6))
    plt.subplot(1, 4, 1)
    plt.title("Model")
    d0 = display_image_sink(mod, fig=fig, vmax=1)

    plt.subplot(1, 4, 2)
    plt.title("Run Image")
    d1 = display_image_sink(img, fig=fig, vmax=5)

    plt.subplot(1, 4, 3)
    plt.title("Cumulative Image")
    d2 = display_image_sink(cum, vmax=300, fig=fig)

    plt.subplot(1, 4, 4)
    plt.title("Cumulative Image - Background")
    d3 = display_image_sink(cumsub, vmax=300, fig=fig)

    plt.show()

    def example_lightcurve():
        # make a lightcurve:
        lc = lightcurve_sink(
            lightcurve_binner(
                smooth_image(
                    sample_from_model(
                        gen_static_model(nmax=1000)
                    ))))
        plt.plot(lc[0], lc[1])
        return lc
