#!/usr/bin/env python3

# run this example with:
#
# python display_event.py <filename>
#
# if no filename is given, a default example file will be used
# containing ~10 events

import logging
import random
import sys

from matplotlib import pyplot as plt
from numpy import ceil, sqrt

from ctapipe.io import event_source
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.visualization import CameraDisplay

logging.basicConfig(level=logging.DEBUG)

fig = plt.figure(figsize=(12, 8))
cmaps = [
    plt.cm.jet, plt.cm.winter, plt.cm.ocean, plt.cm.bone, plt.cm.gist_earth,
    plt.cm.hot, plt.cm.cool, plt.cm.coolwarm
]


def display_event(event):
    """an extremely inefficient display. It creates new instances of
    CameraDisplay for every event and every camera, and also new axes
    for each event. It's hacked, but it works
    """
    print("Displaying... please wait (this is an inefficient implementation)")
    global fig
    ntels = len(event.r0.tels_with_data)
    fig.clear()

    plt.suptitle(f"EVENT {event.r0.event_id}")

    disps = []

    for ii, tel_id in enumerate(event.r0.tels_with_data):
        print(f"\t draw cam {tel_id}...")
        nn = int(ceil(sqrt(ntels)))
        ax = plt.subplot(nn, nn, ii + 1)

        geom = event.inst.subarray.tel[tel_id].camera
        disp = CameraDisplay(geom, ax=ax, title=f"CT{tel_id}")
        disp.pixels.set_antialiaseds(False)
        disp.autoupdate = False
        disp.cmap = random.choice(cmaps)
        chan = 0
        signals = event.r0.tel[tel_id].image[chan].astype(float)
        signals -= signals.mean()
        disp.image = signals
        disp.set_limits_percent(95)
        disp.add_colorbar()
        disps.append(disp)

    return disps


def get_input():
    print("============================================")
    print("n or [enter]    - go to Next event")
    print("d               - Display the event")
    print("p               - Print all event data")
    print("i               - event Info")
    print("s               - save event image")
    print("q               - Quit")
    return input("Choice: ")


if __name__ == '__main__':

    if len(sys.argv) > 1:
        filename = sys.argv.pop(1)
    else:
        filename = get_dataset_path("gamma_test_large.simtel.gz")

    plt.style.use("ggplot")
    plt.show(block=False)

    # loop over events and display menu at each event:
    source = event_source(filename)

    for event in source:

        print(
            "EVENT_ID: ", event.r0.event_id, "TELS: ", event.r0.tels_with_data,
            "MC Energy:", event.mc.energy
        )

        while True:
            response = get_input()
            if response.startswith("d"):
                disps = display_event(event)
                plt.pause(0.1)
            elif response.startswith("p"):
                print("--event-------------------")
                print(event)
                print("--event.r0---------------")
                print(event.r0)
                print("--event.mc----------------")
                print(event.mc)
                print("--event.r0.tel-----------")
                for teldata in event.r0.tel.values():
                    print(teldata)
            elif response == "" or response.startswith("n"):
                break
            elif response.startswith('i'):
                subarray = event.inst.subarray
                for tel_id in sorted(event.r0.tel):
                    for chan in event.r0.tel[tel_id].waveform:
                        npix = len(subarray.tel[tel_id].camera.pix_x)
                        nsamp = event.r0.tel[tel_id].num_samples
                        print(
                            "CT{:4d} ch{} pixels,samples:{}"
                            .format(tel_id, chan, npix, nsamp)
                        )
            elif response.startswith('s'):
                filename = f"event_{event.r0.event_id:010d}.png"
                print("Saving to", filename)
                plt.savefig(filename)

            elif response.startswith('q'):
                break

        if response.startswith('q'):
            break
