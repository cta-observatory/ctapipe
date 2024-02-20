"""
Working with Instrumental Descriptions
======================================

the instrumental description is loaded by the event source, and consists
of a hierarchy of classes in the ctapipe.instrument module, the base of
which is the ``SubarrayDescription``

"""

from astropy.coordinates import SkyCoord

from ctapipe.io import EventSource
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.visualization import CameraDisplay

filename = get_dataset_path("gamma_prod5.simtel.zst")

with EventSource(filename, max_events=1) as source:
    subarray = source.subarray


######################################################################
# the SubarrayDescription:
# ------------------------
#

subarray.info()

######################################################################
subarray.to_table()


######################################################################
# You can also get a table of just the ``OpticsDescriptions``
# (``CameraGeometry`` is more complex and can’t be stored on a single
# table row, so each one can be converted to a table separately)
#

subarray.to_table(kind="optics")


######################################################################
# Make a sub-array with only SC-type telescopes:
#

sc_tels = [tel_id for tel_id, tel in subarray.tel.items() if tel.optics.n_mirrors == 2]
newsub = subarray.select_subarray(sc_tels, name="SCTels")
newsub.info()


######################################################################
# can also do this by using ``Table.group_by``
#


######################################################################
# Explore some of the details of the telescopes
# ---------------------------------------------
#

tel = subarray.tel[1]
tel

######################################################################
tel.optics.mirror_area

######################################################################
tel.optics.n_mirror_tiles

######################################################################
tel.optics.equivalent_focal_length

######################################################################
tel.camera

######################################################################
tel.camera.geometry.pix_x

######################################################################
# %matplotlib inline

CameraDisplay(tel.camera.geometry)

######################################################################
CameraDisplay(subarray.tel[98].camera.geometry)


######################################################################
# Plot the subarray
# -----------------
#
# We’ll make a subarray by telescope type and plot each separately, so
# they appear in different colors. We also calculate the radius using the
# mirror area (and exaggerate it a bit).
#
# This is just for debugging and info, for any “real” use, a
# ``visualization.ArrayDisplay`` should be used
#

subarray.peek()

######################################################################
subarray.footprint


######################################################################
# Get info about the subarray in general
# --------------------------------------
#

subarray.telescope_types

######################################################################
subarray.camera_types

######################################################################
subarray.optics_types


######################################################################
center = SkyCoord("10.0 m", "2.0 m", "0.0 m", frame="groundframe")
coords = subarray.tel_coords  # a flat list of coordinates by tel_index
coords.separation(center)


######################################################################
# Telescope IDs vs Indices
# ------------------------
#
# Note that ``subarray.tel`` is a dict mapped by ``tel_id`` (the
# identifying number of a telescope). It is possible to have telescope
# IDs that do not start at 0, are not contiguouous (e.g. if a subarray is
# selected). Some functions and properties like ``tel_coords`` are numpy
# arrays (not dicts) so they are not mapped to the telescope ID, but
# rather the *index* within this SubarrayDescription. To convert between
# the two concepts you can do:
#

subarray.tel_ids_to_indices([1, 5, 23])


######################################################################
# or you can get the indexing array directly in numpy or dict form:
#

subarray.tel_index_array

######################################################################
subarray.tel_index_array[[1, 5, 23]]

######################################################################
subarray.tel_indices[
    1
]  # this is a dict of tel_id -> tel_index, so we can only do one at once

ids = subarray.get_tel_ids_for_type(subarray.telescope_types[0])
ids

######################################################################
idx = subarray.tel_ids_to_indices(ids)
idx

######################################################################
subarray.tel_coords[idx]


######################################################################
# so, with that method you can quickly get many telescope positions at
# once (the alternative is to use the dict ``positions`` which maps
# ``tel_id`` to a position on the ground
#

subarray.positions[1]
