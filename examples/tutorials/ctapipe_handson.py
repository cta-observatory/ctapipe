"""
Getting Started with ctapipe
============================

This hands-on was initially presented at the Paris CTA Consoritum meeting by K.
Kosack and has been updated since to reflect changes in ctapipe itself.
"""

######################################################################
# Part 1: load and loop over events
# ---------------------------------

import glob

import astropy.units as u
import numpy as np
from ipywidgets import interact
from matplotlib import pyplot as plt

from ctapipe import utils
from ctapipe.calib import CameraCalibrator
from ctapipe.image import hillas_parameters, tailcuts_clean
from ctapipe.io import EventSource, HDF5TableWriter, read_table
from ctapipe.visualization import CameraDisplay

# %matplotlib inline

######################################################################
path = utils.get_dataset_path("gamma_prod5.simtel.zst")

######################################################################
with EventSource(path, max_events=5) as source:
    for event in source:
        print(
            f"{event.count} {event.index.event_id} {event.simulation.shower.energy:.3f} {len(event.tel)}"
        )

######################################################################
event

######################################################################
event.tel.keys()

######################################################################
tel_id = 118
event.tel[tel_id]

######################################################################
with EventSource(path, max_events=5) as source:
    for event in source:
        print(event.count, event.tel.keys())

######################################################################
event.tel[tel_id].r0

######################################################################
r0tel = event.tel[tel_id].r0

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
waveform_sums = r0tel.waveform[0].sum(axis=1)
brightest_pixel = np.argmax(waveform_sums)
print(f"pixel {brightest_pixel} has sum {waveform_sums[brightest_pixel]}")

######################################################################
plt.plot(r0tel.waveform[0, brightest_pixel], label="channel 0 (high-gain)")
plt.plot(r0tel.waveform[1, brightest_pixel], label="channel 1 (low-gain)")
plt.legend()

######################################################################
n_channels, n_pixels, n_samples = r0tel.waveform.shape


@interact(chan=(0, n_channels - 1), pix_id=(0, n_pixels - 1))
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
# This information is provided by the ``SubarrayDescription`` of the event source:

subarray = source.subarray

######################################################################
subarray

######################################################################
subarray.peek()

######################################################################
subarray.to_table()

######################################################################
subarray.tel[tel_id]

######################################################################
subarray.tel[tel_id].camera

######################################################################
subarray.tel[tel_id].optics

######################################################################
tel = subarray.tel[tel_id]

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

# display channel 0, sample 20 (try others like 15)
disp.image = r0tel.waveform[0, :, 20]

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
with EventSource(path, max_events=5) as source:
    for event in source:
        calib(event)  # fills in dl0, and dl1

######################################################################
dl1tel = event.tel[tel_id].dl1

######################################################################
dl1tel.image.shape

######################################################################
dl1tel.peak_time

######################################################################
d = CameraDisplay(tel.camera.geometry, image=dl1tel.image)
d.add_colorbar()

######################################################################
CameraDisplay(tel.camera.geometry, image=dl1tel.peak_time)

######################################################################
# Now for Hillas Parameters
#

image = dl1tel.image
mask = tailcuts_clean(
    tel.camera.geometry,
    image,
    picture_thresh=10,
    boundary_thresh=5,
    keep_isolated_pixels=False,
    min_number_picture_neighbors=2,
)
mask

######################################################################
d = CameraDisplay(tel.camera.geometry, image=image)
d.highlight_pixels(mask)

######################################################################
params = hillas_parameters(tel.camera.geometry[mask], image[mask])
print(params)

######################################################################
disp = CameraDisplay(tel.camera.geometry, image=image)
disp.highlight_pixels(mask, linewidth=2)
disp.add_colorbar()
plt.xlim(params.x.to_value(u.m) - 0.5, params.x.to_value(u.m) + 0.5)
plt.ylim(params.y.to_value(u.m) - 0.5, params.y.to_value(u.m) + 0.5)
disp.overlay_moments(params, color="white", lw=2, n_sigma=2)

######################################################################
# Part 4: Let’s put it all together:
# ----------------------------------
#
# -  loop over events, selecting only telescopes of the same type
#    (e.g. LST_LST_LSTCam)
# -  for each event, apply calibration/trace integration
# -  calculate Hillas parameters
# -  write out all hillas parameters to a file that can be loaded with
#    Pandas
#

######################################################################
# first let’s select only those telescopes with LST_LST_LSTCam

subarray.telescope_types

######################################################################
subarray.get_tel_ids_for_type("LST_LST_LSTCam")

######################################################################
# Now let’s write out program
#

data = utils.get_dataset_path("gamma_prod5.simtel.zst")

######################################################################
with HDF5TableWriter(filename="hillas.h5", group_name="dl1", overwrite=True) as writer:
    with EventSource(data) as source:
        for event in source:
            calib(event)
            for tel_id, tel_event in event.tel.items():
                tel = source.subarray.tel[tel_id]
                image = tel_event.dl1.image

                mask = tailcuts_clean(tel.camera.geometry, image)
                if np.count_nonzero(mask) > 0:
                    params = hillas_parameters(tel.camera.geometry[mask], image[mask])
                    writer.write("hillas", params)

######################################################################
# We can now load in the file we created and plot it
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
glob.glob("*.h5")


######################################################################
hillas = read_table("hillas.h5", "/dl1/hillas")
hillas

######################################################################
fig, axs = plt.subplots(4, 3, figsize=(8, 8), layout="constrained")
axs = axs.ravel()

for col, ax in zip(hillas.colnames, axs):
    ax.hist(hillas[col])
    ax.set_title(col)

######################################################################
# If you do this yourself, chose a larger file to loop over more events to
# get better statistics
#
