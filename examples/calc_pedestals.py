#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
from ctapipe.io.eventsourcefactory import event_source
from ctapipe.utils import get_dataset
from ctapipe.calib import pedestals
import numpy as np

def plot_peds(peds, pedvars):
    """ make a quick plot of the pedestal values"""
    pixid = np.arange(len(peds))
    plt.subplot(1, 2, 1)
    plt.scatter(pixid, peds)
    plt.title("Pedestals for event {}".format(event.r0.event_id))

    plt.subplot(1, 2, 2)
    plt.scatter(pixid, pedvars)
    plt.title("Ped Variances for event {}".format(event.r0.event_id))


if __name__ == '__main__':

    # if a filename is specified, use it, otherwise load sample data
    if len(sys.argv) > 1:
        filename = sys.argv.pop(1)
    else:
        filename = get_dataset("gamma_test.simtel.gz")

    # set a fixed window (for now from samples 20 to the end), which may not
    # be appropriate for all telescopes (this should eventually be
    # chosen based on the telescope type, since some have shorter
    # sample windows:

    start = 15
    end = None  # None means through sample

    # loop over all events, all telescopes and all channels and call
    # the calc_peds function defined above to do some work:
    for event in event_source(filename):
        for telid in event.r0.tels_with_data:
            for chan in range(event.r0.tel[telid].adc_samples.shape[0]):

                print("CT{} chan {}:".format(telid, chan))

                traces = event.r0.tel[telid].adc_samples[chan,...]

                peds, pedvars = pedestals.calc_pedestals_from_traces(traces,
                                                                     start,
                                                                     end)

                print("Number of samples: {}".format(traces.shape[1]))
                print("Calculate over window:({},{})".format(start, end))
                print("PEDS:", peds)
                print("VARS:", pedvars)
                print("-----")

    # as an example, for the final event, let's plot the pedestals and
    # variances: (you could also move this into the for-loop, but
    # would then have tons of plots)

    plot_peds(peds, pedvars)
    plt.show()

    # note with the sample data you'll get a warning about "Wrong
    # number of bytes were read", but that's because the example
    # simtel file is truncated to a small filesize, so the final event
    # is corrupt
