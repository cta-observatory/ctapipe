"""
 first we need a FITS library that can read row-wise efficiently (so
 can't use astropy.io.fits, need a lower-level cfitsio-based system)

* http://stackoverflow.com/questions/22950177/read-fits-binary-table-one-row-at-a-time-using-pyfits
* https://github.com/esheldon/fitsio
* http://sahandsaba.com/understanding-asyncio-node-js-python-3-4.html
"""

import fitsio
import time
import numpy as np
from matplotlib import pyplot as plt
from image_processing_with_generators.py import *

def fits_table_source(url, extension):
    """
    url -- fits file to open (local filename or URL)
    extension -- name or number of extension (HDU)
    """

    header = fitsio.read_header(url, ext=extension)
    n_entries = header["NAXIS2"]

    fitsfile = fitsio.FITS(url, iter_row_buffer=10000)
    table = fitsfile[extension]

    print("--INPUT --------------")
    print(fitsfile)
    print("")
    print(table)
    print("----------------------")

    for ii in range(n_entries):
        row = table[ii]
        yield row.view(np.recarray)

def show_throughput(stream, every=1000):
    count = 0
    tot = 0
    prevtime = 0
    for data in stream:
        yield data
        count += 1; tot += 1
        if count >= every:
            count=0
            curtime=time.time()
            dt = curtime-prevtime
            print("{:10d} events, {:4.1f} evt/s".format( tot, every/dt ))
            prevtime=curtime


def histogram_column(stream, itemname,every=10000):
    fig = plt.figure()
    ax = fig.gca()
    count =0
    bins = np.linspace(-10,10,100)
    hist = np.zeros( len(bins)+1 )
    for data in stream:
        val =data[itemname]
        if np.isfinite(val):
            ii = np.digitize( val, bins )
            hist[ii] += 1

        if (count>every):
            ax.plot( bins, hist[1:], linestyle='steps')
            count = 0

        count += 1
        yield data # pass through



def make_image(stream, bins=(100,100), range=[[-5,5],[-5,5]], nevents=100 ):
    """ Generates an image every `nevents` events """
    image = np.zeros( shape=bins )
    binX = np.linspace( range[0][0], range[0][1], bins[0] )
    binY = np.linspace( range[1][0], range[1][1], bins[1] )
    count = 0
    xpoints = list()
    ypoints = list()

    for data in stream:
        detx = data.DETX[0]
        dety = data.DETY[0]

        # accumulate points for efficiency:
        if (detx > range[0][0] and detx < range[0][1]
            and dety > range[1][0] and dety < range[1][1] ):
            xpoints.append( detx )
            ypoints.append( dety )

        count += 1

        # generate a binned image from the accumulated points:
        if count >= nevents:
            if len(xpoints) > 0:
                ii = np.digitize( xpoints, binX )
                jj = np.digitize( ypoints, binY )
                image[ii,jj] += 1
            yield image.copy() # output the image
            # clear image and data points
            count =0
            image[:] = 0
            xpoints = list()
            ypoints = list()


def event_loop(stream, sleep=None):
    for data in stream:
        if sleep is not None:
            time.sleep(sleep)


def print_event(stream):
    isfirst = True
    for item in stream:
        if isfirst:
            print(item.dtype.names)
            isfirst = False
        print(item)
        yield item


def display_image(stream, pause=0.01):
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    ax.hold(True)

    # draw first image:
    image = next(stream)
    axim = plt.imshow(image, origin="upper", interpolation="nearest")
    plt.show(False)
    plt.draw()
    yield image

    background = fig.canvas.copy_from_bbox(ax.bbox)

    # update the plot on next item:
    for image in stream:
        axim.set_array( image )
        axim.set_clim( 0, image.max()*0.90 )
        plt.pause( pause )
        yield image

if __name__ == '__main__':



#    hh = histogram_column(histogram_column(instream,"DETX"),"AZ")
#    thru = show_throughput(hh)

    instream = fits_table_source( "events.fits","EVENTS")
    im = make_image(show_throughput(instream), bins=(300,300), nevents=500)
    imsum = sum_image(im)

    fig = plt.figure(figsize=(10,7))
    plt.subplot(1,2,1)
    d1 = display_image_sink(smooth_image(im),interval=30, fig=fig)
    plt.subplot(1,2,2)
    d2 = display_image_sink(smooth_image(imsum), interval=30, fig=fig)

    plt.show()


#    for image in imsum:
#        pass
    #event_loop(p, sleep=1)






