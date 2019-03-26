# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from scipy import ndimage
from astropy.io import fits
from astropy.wcs import WCS

__all__ = ['Histogram']


class Histogram:
    """An N-D histogram class with FITS image I/O.

    The output FITS file will contain an ImageHDU datacube and
    associated WCS headers to describe the axes of the histogram.
    Thus, the output files should work correctly in any program
    capable of working with FITS datacubes (like SAOImage DS9).

    Internally, it uses `numpy.histogramdd` to generate the histograms.

    All axes are assumed to be linear, with equally spaced bins
    (otherwise they could not be stored in a FITS image HDU)

    Parameters
    ----------
    nbins: array_like(int)
        list of number of bins for each dimension
    ranges: list(tuple)
        list of (min,max) values for each dimension
    name: str
        name of histogram (will be used as FITS extension name when written
        to a file
    axis_names: list(str)
        name of each axis

    Examples
    --------

    >>> hist = Histogram(nbins=(10,10), ranges=[[-1,1], [-1,1]])
    >>> data = np.random.normal(shape=(2*100)) # make 100 random 2D events
    >>> hist.fill(data)


    Get a point in the histogram (can also get multiple values at once by
    passing an array)

    >>> val = hist.get_value([0.1,-0.5])
    >>> vals = hist.get_value([[0.1,-0.5], [0.9,0.9]])

    Get the full data array and do things with it:

    >>> meanx = hist.data.mean(axis=0)

    Write it to a FITS image file:

    >>> hist.to_fits().writeto("output.fits")

    Read it from FITS image file:

    >>> hist2 = Histogram.from_fits("output.fits")

    """

    def __init__(self, nbins=None, ranges=None, name="Histogram",
                 axis_names=None):
        """ Initialize an unfilled histogram (need to call fill()  put data into it)

        see also
        --------
        The `Histogram.from_fits()` constructor
        """

        self.data = np.zeros(nbins)
        self._bin_lower_edges = None
        self._nbins = np.array([nbins]).flatten()
        self._ranges = np.array(ranges, ndmin=2)
        self.value_scale = None
        self.value_zero = None
        self.name = name
        self._ctypes = None
        self.axis_names = axis_names
        self._numsamples = 0

        # sanity check on inputs:

        if self.ndims < 1:
            raise ValueError("No dimensions specified")
        if self.ndims != len(self._ranges):
            raise ValueError("Dimensions of ranges {} don't match bins {}"
                             .format(len(self._ranges), self.ndims))

        if self.axis_names is not None:  # ensure the array is size ndims
            self.axis_names = np.array(self.axis_names)
            self.axis_names.resize(self.ndims)
        else:
            self.axis_names = [f"axis{x}" for x in range(self.ndims)]

    def __str__(self,):
        return ("Histogram(name='{name}', axes={axnames}, "
                "nbins={nbins}, ranges={ranges})"
                .format(name=self.name, ranges=self._ranges,
                        nbins=self._nbins, axnames=self.axis_names))

    @property
    def bin_lower_edges(self):
        """
        lower edges of bins. The length of the array will be nbins+1,
        since the final edge of the last bin is included for ease of
        use in vector operations
        """
        if self._bin_lower_edges is None:
            self._bin_lower_edges = [np.linspace(self._ranges[ii][0],
                                                 self._ranges[ii][-1],
                                                 self._nbins[ii] + 1)
                                     for ii in range(self.ndims)]
        return self._bin_lower_edges

    @property
    def bins(self):
        return self._nbins

    @property
    def ranges(self):
        return self._ranges

    @property
    def ndims(self):
        return len(self._nbins)

    @property
    def outliers(self):
        """
        returns the number of outlier points (the number of input
        datapoints - the sum of the histogram). This assumes the data
        of the histogram is unmodified (and represents "counts").
        """
        return self._numsamples - self.data.sum()

    def fill(self, datapoints, **kwargs):
        """
        generate a histogram from data points.  Since the results of
        fill() are added to the current histogram, you can call fill()
        multiple times to fill a single Histogram.

        Parameters
        ----------
        datapoints: array_like
            array of N-d points (see `numpy.histogramdd` documentation)
        kwargs:
            any extra options to pass to `numpy.histogramdd` when
            creating the histogram
        """

        hist, __ = np.histogramdd(datapoints, bins=self._nbins,
                                  range=self._ranges, **kwargs)

        self.data += hist
        self._numsamples += len(datapoints)

    def bin_centers(self, index):
        """
        returns array of bin centers for the given index
        """
        return 0.5 * (self.bin_lower_edges[index][1:] +
                      self.bin_lower_edges[index][0:-1])

    def to_fits(self):
        """
        Convert the `Histogram` into an `astropy.io.fits.ImageHDU`,
        suitable for writing to a file.

        Examples
        --------

        >>> myhist.to_fits().writeto("outputfile.fits.gz", overwrite=True)

        """
        ohdu = fits.ImageHDU(data=self.data.transpose())
        ohdu.name = self.name
        ndim = len(self._nbins)

        # the lower-left edge of the first bin is (Xed[0],Yed[0]), which
        # is (0.5,0.5) in FITS pixel coordinates (the center of the bin is
        # at (1.0,1.0))

        # now note that this defines the first pixel in FITS coordinates
        # with the center (1.0,1.0). in integer python coordinates it is [0,0]

        # to transform a world value, you need to subtract 1.0 and round
        # down to get the bin number:
        #       ibin = round( Vpix-1.0 )
        # To get the value of ibin, you need to go the other way:
        #       Vworld[ibin] = wcs.wcs_pix2world( ibin+0.5 )

        for dim in range(ndim):
            width = self._ranges[dim][1] - self._ranges[dim][0]
            num = self._nbins[dim]
            delta = width / float(num)
            bin0pix = 0.5  # lower-left corner of first bin
            bin0coord = self._ranges[dim][0]

            name = self.axis_names[dim]

            ctype = name[0:4] + "-   "
            if self._ctypes is not None:
                ctype = self._ctypes[dim]

            ohdu.header.set("CTYPE%d" % (dim + 1), ctype, name)
            ohdu.header.set("CDELT%d" % (dim + 1), delta)
            ohdu.header.set("CRVAL%d" % (dim + 1), bin0coord)
            ohdu.header.set("CRPIX%d" % (dim + 1), bin0pix)
            ohdu.header.set("CNAME%d" % (dim + 1), self.axis_names[dim])

        if self.value_scale:
            ohdu.header.set("BSCALE", 1.0 / float(self.value_scale))

        if self.value_zero:
            ohdu.header.set("BZERO", float(self.value_zero))

        ohdu.header.set("NSAMP", self._numsamples, "Number of samples "
                                                   "originally filled")

        return ohdu

    @staticmethod
    def from_fits(input_fits):
        """
        Construct a `Histogram` from a previously written FITS histogram file
        or HDU (see `Histogram.to_fits()`)

        Parameters
        ----------
        input_fits: string or astropy.io.fits.ImageHDU
            File or HDU to read into histogram (Should be a FITS HDU
            originally created by `Histogram.to_fits()`, may not work for
            general FITS images)
        """

        hist = Histogram()

        if type(input_fits) == str:
            hdu = fits.open(input_fits)[1]
        else:
            hdu = input_fits

        hist.data = hdu.data.transpose()
        hist._nbins = hist.data.shape

        wcs = WCS(hdu.header)
        ndim = len(hist._nbins)

        edges = []
        hist._ranges = []
        axis_names = []
        hist._ctypes = []

        for dim in range(ndim):
            # note that histogramdd returns edges for 0-N+1 (including
            # the lower edge of the non-existant next bin), so we do
            # the same here to keep things the same
            ax = np.zeros((hist._nbins[dim] + 1, ndim))
            ax[:, dim] = np.arange(hist._nbins[dim] + 1) + 0.5
            edges.append(wcs.wcs_pix2world(ax, 1)[:, dim])
            hist._ranges.append((edges[dim][0], edges[dim][-1]))
            hist._ctypes.append(hdu.header["CTYPE%d" % (dim + 1)])
            if hdu.header.get("CNAME%d" % (dim + 1)):
                axis_names.append(hdu.header["CNAME%d" % (dim + 1)])
            else:
                axis_names.append(hist._ctypes[dim][0:4])

        hist.axis_names = np.array(axis_names)
        hist._bin_lower_edges = edges
        hist._ranges = np.array(hist._ranges)

        if hdu.header.get("BSCALE"):
            hist.value_scale = hdu.header["BSCALE"]

        if hdu.header.get("BSCALE"):
            hist.value_zero = hdu.header["BZERO"]

        if hdu.header.get("NSAMP"):
            hist._numsamples += int(hdu.header["NSAMP"])

        return hist

    def get_value(self, coords, outlier_value=None):
        """ Returns the values of the histogram at the given world
        coordinate(s)

        Parameters
        ----------

        coords: array_like
            array of M coordinates of dimension N (where
            the N is the histogram dimension)
        outlier_value: float or None
          value for outliers, if None, coordinates outside the edges of the
          histogram will be given the edge value

        """
        # if (self._bin_lower_edges == None):
        #     raise ValueError("Histogram is not filled")

        if np.isnan(coords).any():
            raise ValueError("Bad coordinate value")

        world = np.array(coords, ndmin=2)  # at least 2D
        ndims = len(self._nbins)

        bins = np.array([np.digitize(world[:, ii],
                                     self.bin_lower_edges[ii][1:])
                         for ii in range(ndims)])

        maxbin = np.array(self.data.shape)

        # deal with out-of-range values:
        if outlier_value is None:
            # extrapolate (simply for now, just takes edge value)
            bins[bins < 0] = 0
            for ii in range(ndims):
                bins[ii][bins[ii] >= maxbin[ii]] = maxbin[ii] - 1
        else:
            if (bins >= maxbin).any() or (bins < 0).any():
                return outlier_value

        return self.data[tuple(bins)]

    def draw_2d(self, dims=(0, 1), **kwargs):
        """draw the histogram using pcolormesh() (only works for 2D
        histograms currently)

        Parameters
        ----------
        dims: (int,int)
            indices of which dimensions to draw
        kwargs:
            arguments to pass to matplotlib `pcolormesh` command
        """
        from matplotlib import pyplot

        if self.data.ndim < 2:
            raise ValueError("Too few dimensions")

        if len(dims) != 2:
            raise ValueError("dims must be a length-2 integer array")

        pyplot.pcolormesh(self.bin_lower_edges[dims[0]],
                          self.bin_lower_edges[dims[1]],
                          self.data, **kwargs)
        pyplot.title(self.name)
        pyplot.xlabel(self.axis_names[dims[0]])
        pyplot.ylabel(self.axis_names[dims[1]])

    def draw_1d(self, dim=0, **kwargs):
        from matplotlib import pyplot

        # todo fix this to work properly with dim argument!
        pyplot.plot(self.bin_centers(dim), self.data, drawstyle='steps-mid',
                    **kwargs)

    def resample_inplace(self, nbins):
        """
        Change the shape of the histogram using an n-dimensional
        interpolation function (via `ndimage.map_coordinates`).

        Parameters
        ----------
        nbins: tuple of int
            a tuple of the new number of bins to resample_inplace
            this histogram over (e.g. if the original histogram was
            (100,100), setting bins to (200,200) would provide double
            the resolution, with the interviening bins interpolated.
        """

        oldbins = self._nbins
        # iold = np.indices(oldbins)
        inew = np.indices(nbins)

        coords = np.array([inew[X] * (oldbins[X]) / float(nbins[X])
                           for X in range(len(nbins))])

        self._nbins = nbins
        self.data = ndimage.map_coordinates(self.data, coords)
        self._bin_lower_edges = None  # need to be recalculated

    @property
    def hist(self):
        """ for backward compatibility. Use `Histogram.data` for read/write
        access"""
        return self.data
