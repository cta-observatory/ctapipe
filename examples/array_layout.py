from ctapipe.visualization import ArrayDisplay
from ctapipe.utils import datasets
from ctapipe.instrument import InstrumentDescription as ID

from astropy.table import Table
from numpy import ones_like
import matplotlib.pylab as plt

if __name__ == '__main__':

    plt.style.use("ggplot")
    plt.figure(figsize=(9.5, 8.5))

    # load up an example table that has the telescope positions and
    # mirror areas in it:
    arrayfile = datasets.get_path("PROD2_telconfig.fits.gz")
    ID.initialize_telescope(arrayfile)
    tel = ID.Telescope()
    opt = ID.Optics()

    Id = tel.getTelescopeID()
    X = tel.getTelescopePosX()
    Y = tel.getTelescopePosY()
    A = opt.getMirrorArea() * 2  # exaggerate scale a bit
    n = tel.getTelescopeNumber()

    # display the array, and set the color value to 50
    ad = ArrayDisplay(X, Y, A, title="Prod 2 Full Array")
    ad.values = ones_like(X) * 50

    # label them
    for i in range(n):
        name = "CT%i" % Id[i]
        plt.text(X[i], Y[i], name, fontsize=8)

    ad.axes.set_xlim(-1000, 1000)
    ad.axes.set_ylim(-1000, 1000)
    plt.tight_layout()
    plt.show()
