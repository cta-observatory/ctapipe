"""
Table interpolation class, to allow interpolation of 2D images in any number of other 
dimensions. Reads in an interim FITS table as defined below:

Tables organised as a standard FITS image file with each table as an image HDU. 
---------------
First (primary) HDU must contain the following header entries:

CRPIXx, CRVALx, CRDELTAx: Number, position and pixel spacing of the reference pixel in 
the image. Where x is the axis number (1 or 2).

GRIDVALS: Names of values describing the grid points
e.g. GRIDVALS = ALT,AZ
---------------
Each HDU must contain a header entry containing the grid points described by GRIDVALS
e.g. ALT =70
AZI = 0

It is also strongly recommended to include documentation for each interpolation 
dimension containing at least the units. This entry ust begin with DOC, followed by the 
dimension name
e.g. DOCALT = "Altitude of event (deg)"

TODO:
    - Improve error handling
    - Add option for caching nd interpolated value
    - Better deal with edges of phase space
    - Allow non-linear interpolation
"""

from scipy import interpolate
import numpy as np
from astropy.io import fits


class TableInterpolator:
    """
    This is a simple class for loading lookup tables from a fits file and
    interpolating between them
    """

    def __init__(self, filename, verbose=1):
        """
        Initialisation of class to load templates from a file and create the interpolation
        objects

        Parameters
        ----------
        filename: string
            Location of Template file
        verbose: int
            Verbosity level,
            0 = no logging
            1 = File + interpolation point information
            2 = Detailed description of interpolation points
        """
        self.verbose = verbose
        if self.verbose:
            print("Loading lookup tables from", filename)

        grid, bins, template = self.parse_fits_table(filename)
        x_bins, y_bins = bins

        self.interpolator = interpolate.LinearNDInterpolator(grid, template, fill_value=0)
        self.nearest_interpolator = interpolate.NearestNDInterpolator(grid, template)

        self.grid_interp = interpolate.RegularGridInterpolator((x_bins, y_bins),
                                                               np.zeros([x_bins.shape[0],
                                                                         y_bins.shape[
                                                                             0]]),
                                                               method="linear",
                                                               bounds_error=False,
                                                               fill_value=0)

    def parse_fits_table(self, filename):
        """
        Function opens tables contained within fits files and parses them into a format 
        recognisable by the interpolator.

        Parameters
        ----------
        filename: str
            Name of table file

        Returns
        -------
            tuple (grid points, bin centres, images)
        """
        file = fits.open(filename)
        template = list()
        grid = list()

        primary_hdu = file[0].header  # We require first HDU to be primary

        # Below definitions are standard
        ix, iy = primary_hdu["CRPIX2"], primary_hdu["CRPIX1"]
        val_x, val_y = primary_hdu["CRVAL2"], primary_hdu["CRVAL1"]
        print(val_x, val_y)
        delta_x, delta_y = primary_hdu["CRDELTA2"], primary_hdu["CRDELTA1"]
        nbins_x, nbins_y = primary_hdu["NAXIS2"], primary_hdu["NAXIS1"]
        ix *= delta_x
        iy *= delta_y

        x_bins = np.arange(val_x - ix, val_x + (delta_x * nbins_x) - ix, step=delta_x)
        y_bins = np.arange(val_y - iy, val_y + (delta_y * nbins_y) - iy, step=delta_y)
        grid_vals = primary_hdu["GRIDVALS"]
        points = grid_vals.split(",")

        if self.verbose:
            print("Interpolation point source be called in order", points)
        if self.verbose > 1:
            for p in points:
                print(p, ":", primary_hdu["DOC" + p])

        for hdu in file:
            template.append(hdu.data)

            hdu_pt = list()
            for p in points:
                hdu_pt.append(hdu.header[p])
            grid.append(np.array(hdu_pt))

        print(np.array(grid))
        bins = (x_bins, y_bins)
        return grid, bins, template

    def interpolate(self, params, pixel_pos_x, pixel_pos_y):
        """

        Parameters
        ----------
        params: ndarray
            numpy array of interpolation parameters
            currently [energy, impact distance, xmax]
        pixel_pos_x: ndarray
            pixel position in degrees
        pixel_pos_y: ndarray
            pixel position in degrees

        Returns
        -------
        ndarray of expected intensity for all pixel positions given

        """

        image = self.interpolated_image(params)
        self.grid_interp.values = image

        points = np.array([pixel_pos_x, pixel_pos_y])
        return self.grid_interp(points.T)

    def interpolated_image(self, params):
        """
        Function for creating a ful interpolated image template from the interpolation library

        Parameters
        ----------
        params: ndarray
            numpy array of interpolation parameters
            currently [energy, impact distance, xmax]

        Returns
        -------
        ndarray of a single image template

        """
        image = self.interpolator(params)[0]
        if np.isnan(image).all():
            print("Found a NaN", params)
            image = self.nearest_interpolator(params)[0]

        return image
