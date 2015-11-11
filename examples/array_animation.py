import numpy as np
from astropy.table import Table
from ctapipe.utils import datasets
from ctapipe.visualization import ArrayDisplay
from matplotlib import pyplot as plt
from ctapipe.instrument import InstrumentDescription as ID


if __name__ == '__main__':

    plt.style.use("ggplot")
    plt.figure(figsize=(10, 8))

    arrayfile = datasets.get_path("PROD2_telconfig.fits.gz")
    ID.initialize_telescope(arrayfile)
    tel = ID.Telescope()
    opt = ID.Optics()

    adisp = ArrayDisplay(tel.getTelescopePosX(), tel.getTelescopePosY(), opt.getMirrorArea()* 2,
                         title='PROD2 telescopes', autoupdate=False)
    plt.tight_layout()

    values = np.zeros(tel.getTelescopeNumber())

    # do a small animation to show various trigger patterns:

    for ii in range(20):

        # generate a random trigger pattern and integrated intensity:
        ntrig = np.random.poisson(10)
        trigmask = np.random.random_integers(tel.getTelescopeNumber() - 1, size=ntrig)
        values[:] = 0
        values[trigmask] = np.random.uniform(0, 100, size=ntrig)

        # update the display:
        adisp.values = values
        plt.pause(0.5)
