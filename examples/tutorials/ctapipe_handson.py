"""
Getting Started with ctapipe
============================

This hands-on was presented at the Paris CTAO Consoritum meeting (K.
Kosack)

"""


######################################################################
# Part 1: load and loop over data
# -------------------------------
#

import glob

import numpy as np
import pandas as pd
from ipywidgets import interact
from matplotlib import pyplot as plt

from ctapipe import utils
from ctapipe.calib import CameraCalibrator
from ctapipe.image import hillas_parameters, tailcuts_clean
from ctapipe.io import EventSource, HDF5TableWriter
from ctapipe.visualization import CameraDisplay

# %matplotlib inline

######################################################################
path = utils.get_dataset_path("gamma_prod5.simtel.zst")

######################################################################
source = EventSource(path, max_events=5)

for event in source:
    print(event.count, event.index.event_id, event.simulation.shower.energy)

######################################################################
event

######################################################################
event.r1

######################################################################
for event in EventSource(path, max_events=5):
    print(event.count, event.r1.tel.keys())

######################################################################
event.r0.tel[3]

######################################################################
r0tel = event.r0.tel[3]

######################################################################
r0tel.waveform

######################################################################
r0tel.waveform.shape


######################################################################
# note that this is (:math:`N_{channels}`, :math:`N_{pixels}`,
# :math:`N_{samples}`)
#

plt.pcolormesh(r0tel.waveform[0])

######################################################################
brightest_pixel = np.argmax(r0tel.waveform[0].sum(axis=1))
print(f"pixel {brightest_pixel} has sum {r0tel.waveform[0,1535].sum()}")

######################################################################
plt.plot(r0tel.waveform[0, brightest_pixel], label="channel 0 (high-gain)")
plt.plot(r0tel.waveform[1, brightest_pixel], label="channel 1 (low-gain)")
plt.legend()


######################################################################
@interact
def view_waveform(chan=0, pix_id=brightest_pixel):
    plt.plot(r0tel.waveform[chan, pix_id])


######################################################################
# try making this compare 2 waveforms
#


######################################################################
# Part 2: Explore the instrument description
# ------------------------------------------
#
# This is all well and good, but we don’t really know what camera or
# telescope this is… how do we get instrumental description info?
#
# Currently this is returned *inside* the event (it will soon change to be
# separate in next version or so)
#

subarray = source.subarray

######################################################################
subarray

######################################################################
subarray.peek()

######################################################################
subarray.to_table()

######################################################################
subarray.tel[2]

######################################################################
subarray.tel[2].camera

######################################################################
subarray.tel[2].optics

######################################################################
tel = subarray.tel[2]

######################################################################
tel.camera

######################################################################
tel.optics

######################################################################
tel.camera.geometry.pix_x

######################################################################
tel.camera.geometry.to_table()

######################################################################
tel.optics.mirror_area


######################################################################
disp = CameraDisplay(tel.camera.geometry)

######################################################################
disp = CameraDisplay(tel.camera.geometry)
disp.image = r0tel.waveform[
    0, :, 10
]  # display channel 0, sample 0 (try others like 10)


######################################################################
# \*\* aside: \*\* show demo using a CameraDisplay in interactive mode in
# ipython rather than notebook
#


######################################################################
# Part 3: Apply some calibration and trace integration
# ----------------------------------------------------
#


calib = CameraCalibrator(subarray=subarray)

######################################################################
for event in EventSource(path, max_events=5):
    calib(event)  # fills in r1, dl0, and dl1
    print(event.dl1.tel.keys())

######################################################################
event.dl1.tel[3]

######################################################################
dl1tel = event.dl1.tel[3]

######################################################################
dl1tel.image.shape  # note this will be gain-selected in next version, so will be just 1D array of 1855

######################################################################
dl1tel.peak_time

######################################################################
CameraDisplay(tel.camera.geometry, image=dl1tel.image)

######################################################################
CameraDisplay(tel.camera.geometry, image=dl1tel.peak_time)


######################################################################
# Now for Hillas Parameters
#


image = dl1tel.image
mask = tailcuts_clean(tel.camera.geometry, image, picture_thresh=10, boundary_thresh=5)
mask

######################################################################
CameraDisplay(tel.camera.geometry, image=mask)

######################################################################
cleaned = image.copy()
cleaned[~mask] = 0

######################################################################
disp = CameraDisplay(tel.camera.geometry, image=cleaned)
disp.cmap = plt.cm.coolwarm
disp.add_colorbar()
plt.xlim(0.5, 1.0)
plt.ylim(-1.0, 0.0)

######################################################################
params = hillas_parameters(tel.camera.geometry, cleaned)
print(params)

######################################################################
disp = CameraDisplay(tel.camera.geometry, image=cleaned)
disp.cmap = plt.cm.coolwarm
disp.add_colorbar()
plt.xlim(0.5, 1.0)
plt.ylim(-1.0, 0.0)
disp.overlay_moments(params, color="white", lw=2)


######################################################################
# Part 4: Let’s put it all together:
# ----------------------------------
#
# -  loop over events, selecting only telescopes of the same type
#    (e.g. LST:LSTCam)
# -  for each event, apply calibration/trace integration
# -  calculate Hillas parameters
# -  write out all hillas parameters to a file that can be loaded with
#    Pandas
#


######################################################################
# first let’s select only those telescopes with LST:LSTCam
#

subarray.telescope_types

######################################################################
subarray.get_tel_ids_for_type("LST_LST_LSTCam")


######################################################################
# Now let’s write out program
#

data = utils.get_dataset_path("gamma_prod5.simtel.zst")
source = EventSource(data)  # remove the max_events limit to get more stats

######################################################################
for event in source:
    calib(event)

    for tel_id, tel_data in event.dl1.tel.items():
        tel = source.subarray.tel[tel_id]
        mask = tailcuts_clean(tel.camera.geometry, tel_data.image)
        if np.count_nonzero(mask) > 0:
            params = hillas_parameters(tel.camera.geometry[mask], tel_data.image[mask])


######################################################################
with HDF5TableWriter(filename="hillas.h5", group_name="dl1", overwrite=True) as writer:
    source = EventSource(data, allowed_tels=[1, 2, 3, 4], max_events=10)
    for event in source:
        calib(event)

        for tel_id, tel_data in event.dl1.tel.items():
            tel = source.subarray.tel[tel_id]
            mask = tailcuts_clean(tel.camera.geometry, tel_data.image)
            params = hillas_parameters(tel.camera.geometry[mask], tel_data.image[mask])
            writer.write("hillas", params)


######################################################################
# We can now load in the file we created and plot it
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
glob.glob("*.h5")


######################################################################
hillas = pd.read_hdf("hillas.h5", key="/dl1/hillas")
hillas

######################################################################
_ = hillas.hist(figsize=(8, 8))


######################################################################
# If you do this yourself, chose a larger file to loop over more events to
# get better statistics
#
