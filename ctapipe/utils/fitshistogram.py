from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import math

from matplotlib import pyplot
from scipy import ndimage


class Histogram(object):
    """
    A simple N-dimensional histogram class with FITS IO 

    Uses numpy.histogram2d to generate histograms and read and write
    them to FITS files
    """
    
    
    def __init__(self, bins=None, ranges=None, name="Histogram",
                 axisNames=None,initFromFITS=None):
        """ Initialize an unfilled histogram (need to call either
        fill() or specify initFromFITS to put data into it)

        `bins`: array listing binsize for each dimension (this defines
                the dimensions of the histogram)
        `rangess`: array of ranges for each dimension
        `name`: name (use for FITS extension name) 
        """
        
        self.hist = np.zeros(bins)
        self._binLowerEdges= None  #TODO: should be a property, get only
        self._bins = np.array([bins]).flatten()
        self._ranges=np.array(ranges, ndmin=2)
        self.valueScale = None
        self.valueZero = None
        self.name = name
        self._ctypes = None
        self.axisNames = axisNames
        self._numsamples =0

        if (initFromFITS):
            self.loadFromFITS(initFromFITS)

        # sanity check on inputs:
            
        if self.ndims < 1:
            raise ValueError("No dimensions specified")
        if self.ndims != len(self._ranges):
            raise ValueError("Dimensions of ranges {0} don't match bins {1}"\
                                 .format(len(self._ranges), self.ndims))

        if self.axisNames != None: # ensure the array is size ndims
            self.axisNames = np.array(self.axisNames)
            self.axisNames.resize( self.ndims )
        else:
            self.axisNames = []
            for x in range(self.ndims):
                self.axisNames.append("") 
        
        
            


    @property
    def binLowerEdges(self):
        """
        lower edges of bins. The length of the array will be nbins+1,
        since the final edge of the last bin is included for ease of
        use in vector operations
        """
        if (self._binLowerEdges == None):
            self._binLowerEdges = [np.linspace(self._ranges[ii][0],
                                               self._ranges[ii][-1],
                                               self._bins[ii]+1) 
                                   for ii in range(self.ndims)]
        return self._binLowerEdges

    @property
    def bins(self):
        return self._bins
        
    @property
    def ranges(self):
        return self._ranges

    @property
    def ndims(self):
        return len(self._bins)

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
        
        Arguments:
        - `datapoints`: array of points (see numpy.histogramdd() documentation)
        """

        hist, binLowerEdges = np.histogramdd( datapoints, 
                                              bins=self._bins, 
                                              range=self._ranges, **kwargs)
        
        self.hist += hist
        self._numsamples += len(datapoints)

        
    def binCenters(self, index):
        """ 
        returns array of bin centers for the given index
        """
        return 0.5*(self.binLowerEdges[index][1:] + 
                    self.binLowerEdges[index][0:-1])

            


    def asFITS(self):
        """ 
        return A FITS hdu, suitable for writing to disk
        
        to write it, just do 

        myhist.asFITS().writeto("outputfile.fits")

        """
        ohdu = fits.ImageHDU(data=self.hist.transpose())
        ohdu.name = self.name
        ndim = len(self._bins)

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
            num = self._bins[dim]
            delta = width/float(num)
            bin0pix = 0.5 # lower-left corner of first bin
            bin0coord = self._ranges[dim][0]

            name = self.axisNames[dim]

            ctype = name[0:4]+"-   "
            if (self._ctypes != None):
                ctype = self._ctypes[dim]

            ohdu.header.update("CTYPE%d"%(dim+1), ctype,name) 
            ohdu.header.update("CDELT%d"%(dim+1), delta)
            ohdu.header.update("CRVAL%d"%(dim+1), bin0coord)
            ohdu.header.update("CRPIX%d"%(dim+1), bin0pix)
            ohdu.header.update("CNAME%d"%(dim+1), self.axisNames[dim])
            
        if (self.valueScale):
            ohdu.header.update( "BSCALE", 1.0/float(self.valueScale) )

        if (self.valueZero):
            ohdu.header.update( "BZERO", float(self.valueZero) )

        ohdu.header.update("NSAMP", self._numsamples,
                           "Number of samples originally filled" )

        return ohdu


    def loadFromFITS(self, inputFITS):
        """
        load a FITS histogram file
        
        Arguments:
        - `inputfits`: filename or fits HDU
        """

        if (type(inputFITS) == str):
            hdu = fits.open(inputFITS).pop(0)
        else:
            hdu = inputFITS
            
        self.hist = hdu.data.transpose()
        self._bins = self.hist.shape

        wcs = WCS( hdu.header )
        ndim = len(self._bins)
        
        edges = []
        self._ranges = []
        axisnames = []
        self._ctypes = []

        for dim in range(ndim):
            # note that histogramdd returns edges for 0-N+1 (including
            # the lower edge of the non-existant next bin), so we do
            # the same here to keep things the same
            a = np.zeros( (self._bins[dim]+1, ndim ) )
            a[:,dim] = np.arange( self._bins[dim]+1 )+0.5
            edges.append(wcs.wcs_pix2world( a )[:,dim])
            self._ranges.append( (edges[dim][0],edges[dim][-1]) )
            self._ctypes.append(hdu.header["CTYPE%d"%(dim+1)])
            if (hdu.header.get("CNAME%d"%(dim+1))):
                axisnames.append(hdu.header["CNAME%d"%(dim+1)])
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


    def getValue(self, coords, outlierValue=None):
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

        world = np.array( coords, ndmin=2 ) # at least 2D
        ndims = len(self._bins)

        bins = np.array([np.digitize( world[:,ii], 
                                      self.binLowerEdges[ii][1:-1] ) 
                         for ii in range(ndims)])

        maxbin = np.array(self.hist.shape)

        # deal with out-of-range values:
        if (outlierValue==None):
            # extrapolate (simply for now, just takes edge value)
            bins[bins<0] = 0
            for ii in range(ndims):
                bins[ii][bins[ii]>=maxbin[ii]] = maxbin[ii]-1 
        else:
            if (bins>=maxbin).any() or (bins<0).any():
                return outlierValue
        
        return self.hist[tuple(bins)]
                
    def draw2D(self, dims=[0,1], **kwargs):
        """
        draw the histogram using pcolormesh() (only works for 2D histograms currently)
        """
        if self.hist.ndim < 2:
            raise ValueError("Too few dimensions")

        if len(dims) != 2:
            raise ValueError("dims must be a length-2 integer array")

        pyplot.pcolormesh( self.binLowerEdges[dims[0]], 
                           self.binLowerEdges[dims[1]], 
                           self.hist.transpose(),**kwargs )
        pyplot.title( self.name )
        pyplot.xlabel( self.axisNames[dims[0]] )
        pyplot.ylabel( self.axisNames[dims[1]] )
        
    def draw1D(self, dim=0, **kwargs):
        # todo fix this to work properly with dim argument!
        pyplot.plot( self.binCenters(dim), self.hist, drawstyle='steps-mid', 
                     **kwargs)



    def interpolate(self, bins):
        """
        Change the shape of the histogram using an n-dimensional
        interpolation function.

        Arguments:

        - `bins`: a tuple of the new number of bins to interpolate
          this histogram over (e.g. if the original histogram was
          (100,100), setting bins to (200,200) would provide double
          the resolution, with the interviening bins interpolated.
        """
        
        oldbins = self.bins
        iold = np.indices( oldbins )
        inew = np.indices( bins )

        coords = np.array([inew[X]* (oldbins[X])/float(bins[X]) 
                           for X in range(len(bins))])

        self._bins = bins
        self.hist = ndimage.map_coordinates( self.hist, coords )
        self._binLowerEdges = None # need to be recalculated
        
#        raise "error"


        
