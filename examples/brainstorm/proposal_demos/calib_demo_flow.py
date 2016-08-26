"""
Wrap the calib algorithms into a streamable system
"""
import pedestals
from pipe import Pipe as cta_component
import numpy as np
import time
import matplotlib.pyplot as plt
import logging


def gen_fake_raw_images(chunksize=500, chunklimit=None):
    data = dict()
    chunknum = 0
    while True:
        data['image'] = pedestals.make_fake_pedestal_images(nevents=chunksize)
        data['chunknum'] = chunknum
        chunknum += 1
        if chunklimit is not None:
            if chunknum > chunklimit:
                return
        yield data


@cta_component
def calc_and_apply_pedestals(datastream):

    for data in datastream:
        peds, pedvars = pedestals.calc_pedestals(data['image'])
        noisy, dead = pedestals.find_bad_pixels(pedvars)
        pedsub = data['image'] - peds
        pedsub[:, noisy | dead] = 0.0  # set bad pix to 0

        data['peds'] = peds
        data['pedvars'] = pedvars
        data['noisy_pix'] = np.where(noisy)
        data['dead_pix'] = np.where(dead)
        data['image_pedsub'] = pedsub
        yield data


@cta_component
def check_results(datastream,
                  correct_noisy=np.array([10, 100, 740]),
                  correct_dead=np.array([6, 17, 45, 900])):
    n_bad_noisy = 0
    n_bad_dead = 0

    for data in datastream:
        if np.any(data['noisy_pix'] != correct_noisy):
            logging.warning("misidentified noisy pix")
            n_bad_noisy += 1
        if np.any(data['dead_pix'] != correct_dead):
            logging.warning("misidentified dead pix")
            n_bad_dead += 1
        data['n_bad_noisy'] = n_bad_noisy
        data['n_bad_dead'] = n_bad_dead
        yield data


@cta_component
def show_throughput(stream, every=1000, events_per_cycle=1, identifier=""):
    count = 0
    tot = 0
    prevtime = 0
    for data in stream:
        yield data
        count += 1
        tot += 1
        if count >= every:
            count = 0
            curtime = time.time()
            dt = curtime - prevtime
            logging.info("{} {:10d} events, {:4.1f} evt/s"
                         .format(identifier, tot * events_per_cycle,
                                 (every * events_per_cycle) / dt))
            prevtime = curtime


@cta_component
def tee_every(stream, every=1000):
    counter = 0
    for item in stream:
        if counter > every:
            logging.info("-" * 70)
            logging.info(item)
            logging.info("-" * 70)
            counter = 0
        yield item
        counter += 1


@cta_component
def display_pedvars(stream, every=100):
    plt.ion()
    plt.plot(np.arange(10))
    plt.pause(0.1)
    plt.show()
    count = 0
    for data in stream:
        if count % every == 0:
            plt.clf()
            plt.xlabel("Pedestal Variance")
            plt.hist(data['pedvars'], bins=100, range=[0, 100])
            plt.title("Event Chunk {0}".format(data['chunknum']))
            plt.pause(0.001)
        count += 1

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    nevents = 300

    chain = (
        gen_fake_raw_images(chunksize=nevents)
        | calc_and_apply_pedestals
        | check_results
        | show_throughput(every=10, events_per_cycle=nevents,
                          identifier="PED")
        | tee_every(every=100)
        | display_pedvars(every=100)
    )

    # run the chain normally
    for data in chain:
        pass
