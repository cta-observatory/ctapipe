"""
Explore Calibrated Data
=======================
"""

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

import ctapipe
from ctapipe.calib import CameraCalibrator
from ctapipe.image import hillas_parameters, tailcuts_clean
from ctapipe.io import EventSource
from ctapipe.visualization import ArrayDisplay, CameraDisplay

# %matplotlib inline
plt.style.use("ggplot")

print(ctapipe.__version__)
print(ctapipe.__file__)

######################################################################
# Let’s first open a raw event file and get an event out of it:
#

# this is using datasets from ctapipe's test data server
source = EventSource("dataset://gamma_prod5.simtel.zst", max_events=2)

for event in source:
    print(event.index.event_id)

######################################################################
source

######################################################################
event

######################################################################
# to see for which telescopes we have data
print(event.tel.keys())
# get the first one:
tel_id = next(iter(event.tel))

######################################################################
event.tel[tel_id]

######################################################################
calib = CameraCalibrator(subarray=source.subarray)
calib(event)

######################################################################
# Now the *r1*, *dl0* and *dl1* containers are filled in the telescope events
#
# -  **r1**: contains the “r1-calibrated” waveforms, after
#    gain-selection, pedestal subtraciton, and gain-correction
# -  **dl0**: is the same but with optional data volume reduction
#    (some pixels not filled), in this case this is not performed by
#    default, so it is the same as r1
# -  **dl1**: contains the (possibly re-calibrated) waveforms as
#    dl0, but also the time-integrated *image* that has been calculated
#    using a ``ImageExtractor`` (a ``NeighborPeakWindowSum`` by default)
#

for tel_id, tel_event in event.tel.items():
    print("TEL{:03}: {}".format(tel_id, source.subarray.tel[tel_id]))
    print("  - r0  wave shape  : {}".format(tel_event.r0.waveform.shape))
    print("  - r1  wave shape  : {}".format(tel_event.r1.waveform.shape))
    print("  - dl1 image shape : {}".format(tel_event.dl1.image.shape))


######################################################################
# Some image processing:
# ----------------------
#
# Let’s look at the image
#

sub = source.subarray
geometry = sub.tel[tel_id].camera.geometry
image = event.tel[tel_id].dl1.image

######################################################################
disp = CameraDisplay(geometry, image=image)

######################################################################
mask = tailcuts_clean(
    geometry,
    image,
    picture_thresh=10,
    boundary_thresh=5,
    min_number_picture_neighbors=2,
)
disp = CameraDisplay(geometry, image=image)
disp.highlight_pixels(mask)

######################################################################
params = hillas_parameters(geometry[mask], image[mask])
print(params)
params

######################################################################
plt.figure(figsize=(10, 10))
disp = CameraDisplay(geometry, image=image)
disp.add_colorbar()
disp.overlay_moments(params, color="red", lw=3)
disp.highlight_pixels(mask, color="xkcd:light blue", linewidth=5)

plt.xlim(params.x.to_value(u.m) - 0.05, params.x.to_value(u.m) + 0.05)
plt.ylim(params.y.to_value(u.m) - 0.05, params.y.to_value(u.m) + 0.05)

######################################################################
# More complex image processing:
# ------------------------------
#
# Let’s now explore how stereo reconstruction works.
#
# First, look at a summed image from multiple telescopes
#
# For this, we want to use a ``CameraDisplay`` again, but since we can’t
# sum and display images with different cameras, we’ll just sub-select
# images from a particular camera type
#
# These are the telescopes that are in this event:
#

tels_in_event = set(event.tel.keys())
tels_in_event

######################################################################
cam_ids = set(sub.get_tel_ids_for_type("MST_MST_NectarCam"))
cam_ids

######################################################################
cams_in_event = tels_in_event.intersection(cam_ids)
first_tel_id = list(cams_in_event)[0]
tel = sub.tel[first_tel_id]
print("{}s in event: {}".format(tel, cams_in_event))

######################################################################
# Now, let’s sum those images:
#

image_sum = np.zeros(tel.camera.geometry.n_pixels)

for tel_id in cams_in_event:
    image_sum += event.tel[tel_id].dl1.image

######################################################################
# And finally display the sum of those images:

plt.figure(figsize=(8, 8))

disp = CameraDisplay(tel.camera.geometry, image=image_sum)
disp.overlay_moments(params, with_label=False)
plt.title("Sum of {}x {}".format(len(cams_in_event), tel))

######################################################################
# let’s also show which telescopes those were. Note that currently
# ArrayDisplay’s value field is a vector by ``tel_index``.
#

nectarcam_subarray = sub.select_subarray(cam_ids, name="NectarCam")

hit_pattern = np.zeros(shape=nectarcam_subarray.n_tels)
hit_pattern[[nectarcam_subarray.tel_indices[x] for x in cams_in_event]] = 100

plt.set_cmap(plt.cm.Accent)
plt.figure(figsize=(8, 8))

ad = ArrayDisplay(nectarcam_subarray)
ad.values = hit_pattern
ad.add_labels()
