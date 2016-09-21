# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from scipy import ndimage
from astropy.io import fits
from astropy.wcs import WCS

__all__ = ['Histogram']


class Histogram:
    """A simple N-dimensional histogram class that can be written or read
    from a FITS file. 

    The output FITS file will contain an ImageHDU datacube and
    associated WCS headers to describe the axes of the histogram.
    Thus, the output files should work correctly in any program
    capable of working with FITS datacubes (like SAOImage DS9).

    Internally, it uses `numpy.histogramdd` to generate the histograms.

    All axes are assumed to be linear, with equally spaced bins
    (otherwise they could not be stored in a FITS image HDU)

    Parameters
    ----------
    bins: array_like(int)
        list of number of bins for each dimension
    ranges: list(tuple)
        list of (min,max) values for each dimension
    name: str
        name of histogram (will be used as FITS extension name when written
        to a file
    axisNames: list(str)
        name of each axis
    initFromFITS: None or `astropy.io.fits.ImageHDU`
        if not None, construct the Histogram object from the given
        FITS HDU

    """

    def __init__(self, nbins=None, ranges=None, name="Histogram",
                 axisNames=None, initFromFITS=None):
        """ Initialize an unfilled histogram (need to call either
        fill() or specify initFromFITS to put data into it)
        """

        self.hist = np.zeros(nbins)
        self._binLowerEdges = None  # TODO: should be a property, get only
        self._nbins = np.array([nbins]).flatten()
        self._ranges = np.array(ranges, ndmin=2)
        self.valueScale = None
        self.valueZero = None
        self.name = name
        self._ctypes = None
        self.axisNames = axisNames
        self._numsamples = 0

        if (initFromFITS):
            self.from_fits(initFromFITS)

        # sanity check on inputs:

        if self.ndims < 1:
            raise ValueError("No dimensions specified")
        if self.ndims != len(self._ranges):
            raise ValueError("Dimensions of ranges {0} don't match bins {1}"
                             .format(len(self._ranges), self.ndims))

        if self.axisNames is not None:  # ensure the array is size ndims
            self.axisNames = np.array(self.axisNames)
            self.axisNames.resize(self.ndims)
        else:
            self.axisNames = ["axis{}".format(x) for x in range(self.ndims)]
        
    def __str__(self,):
        return ("Histogram(name='{name}', axes={axnames}, "
                "nbins={nbins}, ranges={ranges})"
                .format(name=self.name, ranges=self._ranges,
                        nbins=self._nbins, axnames=self.axisNames))
    
    @property
    def bin_lower_edges(self):
        """
        lower edges of bins. The length of the array will be nbins+1,
        since the final edge of the last bin is included for ease of
        use in vector operations
        """
        if self._binLowerEdges is None:
            self._binLowerEdges = [np.linspace(self._ranges[ii][0],
                                               self._ranges[ii][-1],
                                               self._nbins[ii] + 1)
                                   for ii in range(self.ndims)]
        return self._binLowerEdges

    @property
    def bins(self):
        return self._nbins

    @property
    def ranges(self):
        return self._ranges

    @property
    def ndims(self):
        return len(self._nbins)

    def outliers(self):
        """
        returns the number of outlier points (the number of input
        datapoints - the sum of the histogram). This assumes the data
        of the histogram is unmodified (and is "counts")
        """
        return self._numsamples - self.hist.sum()

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

        hist, binLowerEdges = np.histogramdd(datapoints,
                                             bins=self._nbins,
                                             range=self._ranges, **kwargs)

        self.hist += hist
        self._numsamples += len(datapoints)

    def bin_centers(self, index):
        """ 
        returns array of bin centers for the given index
        """
        return 0.5 * (self.bin_lower_edges[index][1:] +
                      self.bin_lower_edges[index][0:-1])

    def to_fits(self):
        """ 
        return A FITS hdu, suitable for writing to disk

        to write it, just do 

        myhist.to_fits().writeto("outputfile.fits")

        """
        ohdu = fits.ImageHDU(data=self.hist.transpose())
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

            name = self.axisNames[dim]

            ctype = name[0:4] + "-   "
            if (self._ctypes != None):
                ctype = self._ctypes[dim]

            ohdu.header.update("CTYPE%d" % (dim + 1), ctype, name)
            ohdu.header.update("CDELT%d" % (dim + 1), delta)
            ohdu.header.update("CRVAL%d" % (dim + 1), bin0coord)
            ohdu.header.update("CRPIX%d" % (dim + 1), bin0pix)
            ohdu.header.update("CNAME%d" % (dim + 1), self.axisNames[dim])

        if (self.valueScale):
            ohdu.header.update("BSCALE", 1.0 / float(self.valueScale))

        if (self.valueZero):
            ohdu.header.update("BZERO", float(self.valueZero))

        ohdu.header.update("NSAMP", self._numsamples,
                           "Number of samples originally filled")

        return ohdu

    def from_fits(self, inputFITS):
        """
        load a FITS histogram file

        Arguments:
        - `inputfits`: filename or fits HDU
        """

        if (type(inputFITS) == str):
            hdu = fits.open(inputFITS)[1]
        else:
            hdu = inputFITS
        
        self.hist = hdu.data.transpose()
        self._nbins = self.hist.shape

        wcs = WCS(hdu.header)
        ndim = len(self._nbins)

        edges = []
        self._ranges = []
        axisnames = []
        self._ctypes = []

        for dim in range(ndim):
            # note that histogramdd returns edges for 0-N+1 (including
            # the lower edge of the non-existant next bin), so we do
            # the same here to keep things the same
            ax = np.zeros((self._nbins[dim] + 1, ndim))
            ax[:, dim] = np.arange(self._nbins[dim] + 1) + 0.5
            edges.append(wcs.wcs_pix2world(ax, 1)[:, dim])
            self._ranges.append((edges[dim][0], edges[dim][-1]))
            self._ctypes.append(hdu.header["CTYPE%d" % (dim + 1)])
            if (hdu.header.get("CNAME%d" % (dim + 1))):
                axisnames.append(hdu.header["CNAME%d" % (dim + 1)])
            else:
                axisnames.append(self._ctypes[dim][0:4])

        self.axisNames = np.array(axisnames)

        self._binLowerEdges = edges
        self._ranges = np.array(self._ranges)

        if(hdu.header.get("BSCALE")):
            self.valueScale = hdu.header["BSCALE"]

        if(hdu.header.get("BSCALE")):
            self.valueZero = hdu.header["BZERO"]

        if(hdu.header.get("NSAMP")):
            self._numsamples += int(hdu.header["NSAMP"])

    def get_value(self, coords, outlierValue=None):
        """ Returns the values of the histogram at the given world
        coordinate(s) 

        Arguments:

        - `coords`: list/array of M coordinates of dimension N (where
          the N is the histogram dimension)

        - `outlierValue`: value for outliers, if None, coordinates
          outside the edges of the histogram will be given the edge
          value

        """
        # if (self._binLowerEdges == None):
        #     raise ValueError("Histogram is not filled")

        if np.isnan(coords).any():
            raise ValueError("Bad coordinate value")

        world = np.array(coords, ndmin=2)  # at least 2D
        ndims = len(self._nbins)

        bins = np.array([np.digitize(world[:, ii],
                                     self.bin_lower_edges[ii][1:])
                         for ii in range(ndims)])

        maxbin = np.array(self.hist.shape)

        # deal with out-of-range values:
        if (outlierValue == None):
            # extrapolate (simply for now, just takes edge value)
            bins[bins < 0] = 0
            for ii in range(ndims):
                bins[ii][bins[ii] >= maxbin[ii]] = maxbin[ii] - 1
        else:
            if (bins >= maxbin).any() or (bins < 0).any():
                return outlierValue

        return self.hist[tuple(bins)]

    def draw_2d(self, dims=[0, 1], **kwargs):
        """draw the histogram using pcolormesh() (only works for 2D
        histograms currently)

        Parameters
        ----------
        self: type
            description
        dims: (int,int)
            indices of which dimensions to draw
        kwargs: 
            arguments to pass to matplotlib `pcolormesh` command
        """
        from matplotlib import pyplot

        if self.hist.ndim < 2:
            raise ValueError("Too few dimensions")

        if len(dims) != 2:
            raise ValueError("dims must be a length-2 integer array")

        pyplot.pcolormesh(self.bin_lower_edges[dims[0]],
                          self.bin_lower_edges[dims[1]],
                          self.hist, **kwargs)
        pyplot.title(self.name)
        pyplot.xlabel(self.axisNames[dims[0]])
        pyplot.ylabel(self.axisNames[dims[1]])

    def draw_1d(self, dim=0, **kwargs):
        from matplotlib import pyplot

        # todo fix this to work properly with dim argument!
        pyplot.plot(self.bin_centers(dim), self.hist, drawstyle='steps-mid',
                    **kwargs)

    def interpolate(self, nbins):
        """
        Change the shape of the histogram using an n-dimensional
        interpolation function.

        Arguments:

        - `bins`: a tuple of the new number of bins to interpolate
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
        self.hist = ndimage.map_coordinates(self.hist, coords)
        self._binLowerEdges = None  # need to be recalculated
