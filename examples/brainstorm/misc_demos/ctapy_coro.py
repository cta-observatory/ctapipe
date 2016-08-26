import fitsio
import time
import numpy as np
from matplotlib import pyplot as plt
from image_processing_with_generators import *


def coroutine(func):
    """decorator to automatically "prime" the coroutine (e.g. to send
    None the first time, to get the init parts of the coroutine to
    execute)

    """
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        cr.send(None)
        return cr
    return start


def fits_table_driver(output, url, extension):
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
        data = table[ii]
        output.send(dict(event=data))


@coroutine
def print_data(output):
    while True:
        data = (yield)
        print(data)
        if output:
            output.send(data)


@coroutine
def show_throughput(output, every=1000, multfactor=1):
    count = 0
    tot = 0
    prevtime = 0
    while True:
        data = (yield)
        count += 1
        tot += 1
        if count >= every:
            count = 0
            curtime = time.time()
            dt = curtime - prevtime
            print(
                "{:10d} events, {:4.1f} evt/s".format(tot, every * multfactor / dt))
            prevtime = curtime
        output.send(data)


@coroutine
def split(output, output2, every=1000):
    """ send data to output, but every N events send also to output2 """
    count = 0
    while True:
        data = (yield)
        if count % every == 0:
            output2.send(data)
        count += 1
        output.send(data)


@coroutine
def writer():
    while True:
        data = (yield)

if __name__ == '__main__':

    # Options are just kwargs for each coroutine read them from
    # commandline+conffile and pass the sub-dictionary to each module
    #
    #
    # [show_throughput]
    #     every = 1000
    #
    # opts = read_opts_from_commandline()
    # show_throughput( sp, **opts['show_throughput'] )
    #
    # Modules should contain:
    # - main coroutine that does the processing
    # - an input simulator
    # - a test case that chains the innput simulator to the coroutine
    # - any "internal" support functions
    #
    # Coroutines should look like:
    #    @cta.component
    #    co_stuff( output1, output2,.., opt1=val1, opt2=val2, ...):
    #        data = yield
    #        output1.send(data)
    # (e.g. args = outputs, kwargs = options)
    #  the decorator could just be a standard primer, or could
    #  do something fancier like register the function to be primed in
    # a list, and then can call prime()
    #

    sp = split(writer(),  print_data(None), every=50000)
    chain = show_throughput(sp, every=10000)

    fits_table_driver(chain,  "events.fits", "EVENTS")
