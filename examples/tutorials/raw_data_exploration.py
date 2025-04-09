"""
Exploring Raw Data
==================

Here are just some very simple examples of going through and inspecting
the raw data, and making some plots using ``ctapipe``. The data explored
here are *raw Monte Carlo* data, which is Data Level “R0” in CTAO
terminology (e.g. it is before any processing that would happen inside a
Camera or off-line)

"""


######################################################################
# Setup:
#

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from ctapipe.io import EventSource
from ctapipe.utils import get_dataset_path
from ctapipe.visualization import CameraDisplay

# %matplotlib inline


######################################################################
# To read SimTelArray format data, ctapipe uses the ``pyeventio`` library
# (which is installed automatically along with ctapipe). The following
# lines however will load any data known to ctapipe (multiple
# ``EventSources`` are implemented, and chosen automatically based on the
# type of the input file.
#
# All data access first starts with an ``EventSource``, and here we use a
# helper function ``event_source`` that constructs one. The resulting
# ``source`` object can be iterated over like a list of events. We also
# here use an ``EventSeeker`` which provides random-access to the source
# (by seeking to the given event ID or number)
#

source = EventSource(get_dataset_path("gamma_prod5.simtel.zst"), max_events=5)


######################################################################
# Explore the contents of an event
# --------------------------------
#
# note that the R0 level is the raw data that comes out of a camera, and
# also the lowest level of monte-carlo data.
#

# so we can advance through events one-by-one
event_iterator = iter(source)

event = next(event_iterator)


######################################################################
# the event is just a class with a bunch of data items in it. You can see
# a more compact representation via:
#

event.r0


######################################################################
# printing the event structure, will currently print the value all items
# under it (so you get a lot of output if you print a high-level
# container):
#

print(event.simulation.shower)

######################################################################
print(event.r0.tel.keys())


######################################################################
# note that the event has 3 telescopes in it: Let’s try the next one:
#

event = next(event_iterator)
print(event.r0.tel.keys())


######################################################################
# now, we have a larger event with many telescopes… Let’s look at one of
# them:
#

teldata = event.r0.tel[26]
print(teldata)
teldata


######################################################################
# Note that some values are unit quantities (``astropy.units.Quantity``)
# or angular quantities (``astropy.coordinates.Angle``), and you can
# easily manipulate them:
#

event.simulation.shower.energy

######################################################################
event.simulation.shower.energy.to("GeV")

######################################################################
event.simulation.shower.energy.to("J")

######################################################################
event.simulation.shower.alt

######################################################################
print("Altitude in degrees:", event.simulation.shower.alt.deg)


######################################################################
# Look for signal pixels in a camera
# ----------------------------------
#
# again, ``event.r0.tel[x]`` contains a data structure for the telescope
# data, with some fields like ``waveform``.
#
# Let’s make a 2D plot of the sample data (sample vs pixel), so we can see
# if we see which pixels contain Cherenkov light signals:
#

plt.pcolormesh(teldata.waveform[0])  # note the [0] is for channel 0
plt.colorbar()
plt.xlabel("sample number")
plt.ylabel("Pixel_id")


######################################################################
# Let’s zoom in to see if we can identify the pixels that have the
# Cherenkov signal in them
#

plt.pcolormesh(teldata.waveform[0])
plt.colorbar()
plt.ylim(700, 750)
plt.xlabel("sample number")
plt.ylabel("pixel_id")
print("waveform[0] is an array of shape (N_pix,N_slice) =", teldata.waveform[0].shape)


######################################################################
# Now we can really see that some pixels have a signal in them!
#
# Lets look at a 1D plot of pixel 270 in channel 0 and see the signal:
#

trace = teldata.waveform[0][719]
plt.plot(trace, drawstyle="steps")


######################################################################
# Great! It looks like a *standard Cherenkov signal*!
#
# Let’s take a look at several traces to see if the peaks area aligned:
#

for pix_id in range(718, 723):
    plt.plot(
        teldata.waveform[0][pix_id], label="pix {}".format(pix_id), drawstyle="steps"
    )
plt.legend()


######################################################################
# Look at the time trace from a Camera Pixel
# ------------------------------------------
#
# ``ctapipe.calib.camera`` includes classes for doing automatic trace
# integration with many methods, but before using that, let’s just try to
# do something simple!
#
# Let’s define the integration windows first: By eye, they seem to be
# reaonsable from sample 8 to 13 for signal, and 20 to 29 for pedestal
# (which we define as the sum of all noise: NSB + electronic)
#

for pix_id in range(718, 723):
    plt.plot(teldata.waveform[0][pix_id], "+-")
plt.fill_betweenx([0, 1600], 19, 24, color="red", alpha=0.3, label="Ped window")
plt.fill_betweenx([0, 1600], 5, 9, color="green", alpha=0.3, label="Signal window")
plt.legend()


######################################################################
# Do a very simplisitic trace analysis
# ------------------------------------
#
# Now, let’s for example calculate a signal and background in a the fixed
# windows we defined for this single event. Note we are ignoring the fact
# that cameras have 2 gains, and just using a single gain (channel 0,
# which is the high-gain channel):
#

data = teldata.waveform[0]
peds = data[:, 19:24].mean(axis=1)  # mean of samples 20 to 29 for all pixels
sums = data[:, 5:9].sum(axis=1) / (13 - 8)  # simple sum integration

######################################################################
phist = plt.hist(peds, bins=50, range=[0, 150])
plt.title("Pedestal Distribution of all pixels for a single event")


######################################################################
# let’s now take a look at the pedestal-subtracted sums and a
# pedestal-subtracted signal:
#

plt.plot(sums - peds)
plt.xlabel("pixel id")
plt.ylabel("Pedestal-subtracted Signal")


######################################################################
# Now, we can clearly see that the signal is centered at 0 where there is
# no Cherenkov light, and we can also clearly see the shower around pixel
# 250.
#

# we can also subtract the pedestals from the traces themselves, which would be needed to compare peaks properly
for ii in range(270, 280):
    plt.plot(data[ii] - peds[ii], drawstyle="steps", label="pix{}".format(ii))
plt.legend()


######################################################################
# Camera Displays
# ---------------
#
# It’s of course much easier to see the signal if we plot it in 2D with
# correct pixel positions!
#
#    note: the instrument data model is not fully implemented, so there is
#    not a good way to load all the camera information (right now it is
#    hacked into the ``inst`` sub-container that is read from the
#    Monte-Carlo file)
#

camgeom = source.subarray.tel[24].camera.geometry

######################################################################
title = "CT24, run {} event {} ped-sub".format(event.index.obs_id, event.index.event_id)
disp = CameraDisplay(camgeom, title=title)
disp.image = sums - peds
disp.cmap = plt.cm.RdBu_r
disp.add_colorbar()
disp.set_limits_percent(95)  # autoscale


######################################################################
# It looks like a nice signal! We have plotted our pedestal-subtracted
# trace integral, and see the shower clearly!
#
# Let’s look at all telescopes:
#
#    note we plot here the raw signal, since we have not calculated the
#    pedestals for each)
#

for tel in event.r0.tel.keys():
    plt.figure()
    camgeom = source.subarray.tel[tel].camera.geometry
    title = "CT{}, run {} event {}".format(
        tel, event.index.obs_id, event.index.event_id
    )
    disp = CameraDisplay(camgeom, title=title)
    disp.image = event.r0.tel[tel].waveform[0].sum(axis=1)
    disp.cmap = plt.cm.RdBu_r
    disp.add_colorbar()
    disp.set_limits_percent(95)


######################################################################
# some signal processing…
# -----------------------
#
# Let’s try to detect the peak using the scipy.signal package:
# https://docs.scipy.org/doc/scipy/reference/signal.html
#


pix_ids = np.arange(len(data))
has_signal = sums > 300

widths = np.array(
    [
        8,
    ]
)  # peak widths to search for (let's fix it at 8 samples, about the width of the peak)
peaks = [signal.find_peaks_cwt(trace, widths) for trace in data[has_signal]]

for p, s in zip(pix_ids[has_signal], peaks):
    print("pix{} has peaks at sample {}".format(p, s))
    plt.plot(data[p], drawstyle="steps-mid")
    plt.scatter(np.array(s), data[p, s])


######################################################################
# clearly the signal needs to be filtered first, or an appropriate wavelet
# used, but the idea is nice
#
