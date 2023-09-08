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
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.visualization import ArrayDisplay, CameraDisplay

# %matplotlib inline
plt.style.use("ggplot")

print(ctapipe.__version__)
print(ctapipe.__file__)


######################################################################
# Let’s first open a raw event file and get an event out of it:
#

filename = get_dataset_path("gamma_prod5.simtel.zst")
source = EventSource(filename, max_events=2)

for event in source:
    print(event.index.event_id)

######################################################################
filename

######################################################################
source

######################################################################
event

######################################################################
print(event.r1)


######################################################################
# Perform basic calibration:
# --------------------------
#
# Here we will use a ``CameraCalibrator`` which is just a simple wrapper
# that runs the three calibraraton and trace-integration phases of the
# pipeline, taking the data from levels:
#
# **R0** → **R1** → **DL0** → **DL1**
#
# You could of course do these each separately, by using the classes
# ``R1Calibrator``, ``DL0Reducer``, and ``DL1Calibrator``. Note that we
# have not specified any configuration to the ``CameraCalibrator``, so it
# will be using the default algorithms and thresholds, other than
# specifying that the product is a “HESSIOR1Calibrator” (hopefully in the
# near future that will be automatic).
#


calib = CameraCalibrator(subarray=source.subarray)
calib(event)


######################################################################
# Now the *r1*, *dl0* and *dl1* containers are filled in the event
#
# -  **r1.tel[x]**: contains the “r1-calibrated” waveforms, after
#    gain-selection, pedestal subtraciton, and gain-correction
# -  **dl0.tel[x]**: is the same but with optional data volume reduction
#    (some pixels not filled), in this case this is not performed by
#    default, so it is the same as r1
# -  **dl1.tel[x]**: contains the (possibly re-calibrated) waveforms as
#    dl0, but also the time-integrated *image* that has been calculated
#    using a ``ImageExtractor`` (a ``NeighborPeakWindowSum`` by default)
#

for tel_id in event.dl1.tel:
    print("TEL{:03}: {}".format(tel_id, source.subarray.tel[tel_id]))
    print("  - r0  wave shape  : {}".format(event.r0.tel[tel_id].waveform.shape))
    print("  - r1  wave shape  : {}".format(event.r1.tel[tel_id].waveform.shape))
    print("  - dl1 image shape : {}".format(event.dl1.tel[tel_id].image.shape))


######################################################################
# Some image processing:
# ----------------------
#
# Let’s look at the image
#


tel_id = sorted(event.r1.tel.keys())[1]
sub = source.subarray
geometry = sub.tel[tel_id].camera.geometry
image = event.dl1.tel[tel_id].image

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
cleaned = image.copy()
cleaned[~mask] = 0
disp = CameraDisplay(geometry, image=cleaned)

######################################################################
params = hillas_parameters(geometry, cleaned)
print(params)
params

######################################################################
params = hillas_parameters(geometry, cleaned)

plt.figure(figsize=(10, 10))
disp = CameraDisplay(geometry, image=image)
disp.add_colorbar()
disp.overlay_moments(params, color="red", lw=3)
disp.highlight_pixels(mask, color="white", alpha=0.3, linewidth=2)

plt.xlim(params.x.to_value(u.m) - 0.5, params.x.to_value(u.m) + 0.5)
plt.ylim(params.y.to_value(u.m) - 0.5, params.y.to_value(u.m) + 0.5)

######################################################################
source.metadata


######################################################################
# More complex image processing:
# ------------------------------
#
# Let’s now explore how stereo reconstruction works.
#
# first, look at a summed image from multiple telescopes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# For this, we want to use a ``CameraDisplay`` again, but since we can’t
# sum and display images with different cameras, we’ll just sub-select
# images from a particular camera type
#
# These are the telescopes that are in this event:
#

tels_in_event = set(
    event.dl1.tel.keys()
)  # use a set here, so we can intersect it later
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

image_sum = np.zeros_like(
    tel.camera.geometry.pix_x.value
)  # just make an array of 0's in the same shape as the camera

for tel_id in cams_in_event:
    image_sum += event.dl1.tel[tel_id].image


######################################################################
# And finally display the sum of those images
#

plt.figure(figsize=(8, 8))

disp = CameraDisplay(tel.camera.geometry, image=image_sum)
disp.overlay_moments(params, with_label=False)
plt.title("Sum of {}x {}".format(len(cams_in_event), tel))


######################################################################
# let’s also show which telescopes those were. Note that currently
# ArrayDisplay’s value field is a vector by ``tel_index``, not ``tel_id``,
# so we have to convert to a tel_index. (this may change in a future
# version to be more user-friendly)
#


nectarcam_subarray = sub.select_subarray(cam_ids, name="NectarCam")

hit_pattern = np.zeros(shape=nectarcam_subarray.n_tels)
hit_pattern[[nectarcam_subarray.tel_indices[x] for x in cams_in_event]] = 100

plt.set_cmap(plt.cm.Accent)
plt.figure(figsize=(8, 8))

ad = ArrayDisplay(nectarcam_subarray)
ad.values = hit_pattern
ad.add_labels()
