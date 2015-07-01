"""Examples of N-dimensional interpolation, which could be used for
example to read and interpolate an lookup table or IRF.

In this example, we load a sample energy reconstruction lookup-table
from a FITS file (in this case it is only in 2D: SIZE vs
IMPACT-DISTANCE.

"""
import matplotlib.pylab as plt
import numpy as np
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

from ctapipe.utils.datasets import get_path
from ctapipe.utils import Histogram


if __name__ == '__main__':

    testfile = get_path("hess_ct001_energylut.fits.gz")

    energy_hdu = fits.open(testfile)['MEAN']
    energy_count_hdu = fits.open(testfile)['COUNT']
    energy_table = Histogram(initFromFITS=energy_hdu)
    energy_count_table = Histogram(initFromFITS=energy_hdu)

    # Now, construct an interpolator that we can use to get values at
    # any point:
    centers = [energy_table.binCenters(ii) for ii in range(energy_table.ndims)]
    energy_lookup = RegularGridInterpolator( centers, energy_table.hist )
    energy_count_lookup = RegularGridInterpolator( centers,
                                                    energy_count_table.hist )

    # energy_lookup is now just a continuous function of log(SIZE),
    # DISTANCE in m.  We can now plot some curves from the
    # interpolator.  The errorbars will be the sqrt(Ncounts) for each
    # interpolated position, just as an example
    
    lsize = np.linspace(1.5, 5.0,100)
    dists = np.linspace(10,80,5)

    plt.loglog()
    plt.title("Variation of energy with size and impact distance")
    plt.xlabel("SIZE (P.E.)")
    plt.ylabel("ENERGY (TeV)")
    
    for dist in dists:
        print(dist)
        plt.errorbar( 10**lsize, 10**energy_lookup((lsize, dist)),
                      yerr=sqrt(energy_count_lookup((lsize,dist))),
                      label="DIST={:.1f} m".format(dist))

    plt.legend(loc="best")

    # note that the LUT we used is does not have very high statistics,
    # so the interpolation starts to be affected by noise at the high
    # end. In a real case, we would want to use a table that has been
    # sanitized (smoothed and extrapolated)
