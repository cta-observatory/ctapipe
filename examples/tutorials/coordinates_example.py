"""
Coordinates usage in ctapipe
============================

"""

import copy

import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from ctapipe.coordinates import CameraFrame, NominalFrame, TelescopeFrame
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import EventSource
from ctapipe.utils import get_dataset_path
from ctapipe.visualization import CameraDisplay

# %matplotlib inline


# make plots and fonts larger
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 16


######################################################################
# Open test dataset
# -----------------
#

filename = get_dataset_path("gamma_prod5.simtel.zst")
source = EventSource(filename)

events = [copy.deepcopy(event) for event in source]
event = events[4]

layout = set(source.subarray.tel_ids)


######################################################################
# Choose event with LST
# ~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# This ensures that the telescope is not “parked” (as it would be in an
# event where it is not triggered) but is actually pointing to a source.
#

print(f"Telescope with data: {event.r1.tel.keys()}")
tel_id = 3


######################################################################
# AltAz
# -----
#
# See `Astropy Docs on
# AltAz <https://docs.astropy.org/en/stable/api/astropy.coordinates.AltAz.html>`__.
#
# Pointing direction of telescopes or the origin of a simulated shower are
# described in the ``AltAz`` frame. This is a local, angular coordinate
# frame, with angles ``altitude`` and ``azimuth``. Altitude is the
# measured from the Horizon (0°) to the Zenith (90°). For the azimuth,
# there are different conventions. In Astropy und thus ctapipe, Azimuth is
# oriented East of North (i.e., N=0°, E=90°).
#


obstime = Time("2013-11-01T03:00")
location = EarthLocation.of_site("Roque de los Muchachos")

altaz = AltAz(location=location, obstime=obstime)

array_pointing = SkyCoord(
    alt=event.pointing.array_azimuth,
    az=event.pointing.array_altitude,
    frame=altaz,
)

print(array_pointing)


######################################################################
# CameraFrame
# -----------
#
# Camera coordinate frame.
#
# The camera frame is a 2d cartesian frame, describing position of objects
# in the focal plane of the telescope.
#
# The frame is defined as in H.E.S.S., starting at the horizon, the
# telescope is pointed to magnetic north in azimuth and then up to zenith.
#
# Now, x points north and y points west, so in this orientation, the
# camera coordinates line up with the CORSIKA ground coordinate system.
#
# MAGIC and FACT use a different camera coordinate system: Standing at the
# dish, looking at the camera, x points right, y points up. To transform
# MAGIC/FACT to ctapipe, do x’ = -y, y’ = -x.
#
# **Typical usage**: Position of pixels in the focal plane.
#

geometry = source.subarray.tel[tel_id].camera.geometry
pix_x = geometry.pix_x
pix_y = geometry.pix_y
focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length

######################################################################
telescope_pointing = SkyCoord(
    alt=event.pointing.tel[tel_id].altitude,
    az=event.pointing.tel[tel_id].azimuth,
    frame=altaz,
)

camera_frame = CameraFrame(
    focal_length=focal_length,
    rotation=0 * u.deg,
    telescope_pointing=telescope_pointing,
)

cam_coords = SkyCoord(x=pix_x, y=pix_y, frame=camera_frame)

print(cam_coords)

######################################################################
plt.scatter(cam_coords.x, cam_coords.y)
plt.title(f"Camera type: {geometry.name}")
plt.xlabel(f"x / {cam_coords.x.unit}")
plt.ylabel(f"y / {cam_coords.y.unit}")
plt.axis("square")


######################################################################
# The implementation of the coordinate system with astropy makes it easier
# to use time of the observation and location of the observing site, to
# understand, for example which stars are visible during a certain night
# and how they might be visible in the camera.
#


location = EarthLocation.of_site("Roque de los Muchachos")
obstime = Time("2018-11-01T04:00")

crab = SkyCoord.from_name("crab nebula")

altaz = AltAz(location=location, obstime=obstime)

pointing = crab.transform_to(altaz)

camera_frame = CameraFrame(
    telescope_pointing=pointing,
    focal_length=focal_length,
    obstime=obstime,
    location=location,
)


subarray = SubarrayDescription.read("dataset://gamma_prod5.simtel.zst")
cam = subarray.tel[1].camera.geometry
fig, ax = plt.subplots()
display = CameraDisplay(cam, ax=ax)

ax.set_title(
    f"La Palma, {obstime}, az={pointing.az.deg:.1f}°, zenith={pointing.zen.deg:.1f}°, camera={geometry.name}"
)

for i, name in enumerate(["crab nebula", "o tau", "zet tau"]):
    star = SkyCoord.from_name(name)
    star_cam = star.transform_to(camera_frame)

    x = star_cam.x.to_value(u.m)
    y = star_cam.y.to_value(u.m)

    ax.plot(x, y, marker="*", color=f"C{i}")
    ax.annotate(
        name,
        xy=(x, y),
        xytext=(5, 5),
        textcoords="offset points",
        color=f"C{i}",
    )

plt.show()


######################################################################
# TelescopeFrame
# --------------
#
# Telescope coordinate frame. A ``Frame`` using a
# ``UnitSphericalRepresentation``.
#
# This is basically the same as a ``HorizonCoordinate``, but the origin is
# at the telescope’s pointing direction. This is what astropy calls a
# ``SkyOffsetFrame``.
#
# The axis of the telescope frame, ``fov_lon`` and ``fov_lat``, are
# aligned with the horizontal system’s azimuth and altitude respectively.
#
# Pointing corrections should applied to the transformation between this
# frame and the camera frame.
#

telescope_frame = TelescopeFrame(
    telescope_pointing=pointing,
    obstime=pointing.obstime,
    location=pointing.location,
)
telescope_coords = cam_coords.transform_to(telescope_frame)

######################################################################
wrap_angle = telescope_pointing.az + 180 * u.deg

plt.axis("equal")
plt.scatter(
    telescope_coords.fov_lon.deg, telescope_coords.fov_lat.deg, alpha=0.2, color="gray"
)


for i, name in enumerate(["crab nebula", "o tau", "zet tau"]):
    star = SkyCoord.from_name(name)
    star_tel = star.transform_to(telescope_frame)

    plt.plot(star_tel.fov_lon.deg, star_tel.fov_lat.deg, "*", ms=10)
    plt.annotate(
        name,
        xy=(star_tel.fov_lon.deg, star_tel.fov_lat.deg),
        xytext=(5, 5),
        textcoords="offset points",
        color=f"C{i}",
    )

plt.xlabel("fov_lon / {}".format(telescope_coords.altaz.az.unit))
plt.ylabel("fov_lat / {}".format(telescope_coords.altaz.alt.unit))


######################################################################
# NominalFrame
# ------------
#
# Nominal coordinate frame. A Frame using a
# ``UnitSphericalRepresentation``. This is basically the same as a
# ``HorizonCoordinate``, but the origin is at an arbitrary position in the
# sky. This is what astropy calls a ``SkyOffsetFrame`` If the telescopes
# are in divergent pointing, this ``Frame`` can be used to transform to a
# common system. - 2D reconstruction (``HillasIntersector``) is performed
# in this frame - 3D reconstruction (``HillasReconstructor``) doesn’t need
# this frame
#


######################################################################
# Let’s play a bit with 3 LSTs with divergent pointing
#

location = EarthLocation.of_site("Roque de los Muchachos")
obstime = Time("2018-11-01T02:00")
altaz = AltAz(location=location, obstime=obstime)

crab = SkyCoord.from_name("crab nebula")

# let's observe crab
array_pointing = crab.transform_to(altaz)


# let the telescopes point to different positions
alt_offsets = u.Quantity([1, -1, -1], u.deg)
az_offsets = u.Quantity([0, -2, +2], u.deg)


tel_pointings = SkyCoord(
    alt=array_pointing.alt + alt_offsets,
    az=array_pointing.az + az_offsets,
    frame=altaz,
)

camera_frames = CameraFrame(
    telescope_pointing=tel_pointings,  # multiple pointings, so we get multiple frames
    focal_length=focal_length,
    obstime=obstime,
    location=location,
)

nom_frame = NominalFrame(origin=array_pointing, obstime=obstime, location=location)

######################################################################
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_aspect(1)

for i in range(3):
    cam_coord = SkyCoord(x=pix_x, y=pix_y, frame=camera_frames[i])
    nom_coord = cam_coord.transform_to(nom_frame)

    ax.scatter(
        x=nom_coord.fov_lon.deg,
        y=nom_coord.fov_lat.deg,
        label=f"Telescope {i + 1}",
        s=30,
        alpha=0.15,
    )


for i, name in enumerate(["Crab", "o tau", "zet tau"]):
    s = SkyCoord.from_name(name)
    s_nom = s.transform_to(nom_frame)
    ax.plot(
        s_nom.fov_lon.deg,
        s_nom.fov_lat.deg,
        "*",
        ms=10,
    )
    ax.annotate(
        name,
        xy=(s_nom.fov_lon.deg, s_nom.fov_lat.deg),
        xytext=(5, 5),
        textcoords="offset points",
        color=f"C{i}",
    )


ax.set_xlabel("fov_lon / deg")
ax.set_ylabel("fov_lat / deg")

ax.legend()
plt.show()


######################################################################
# GroundFrame
# -----------
#
# Ground coordinate frame. The ground coordinate frame is a simple
# cartesian frame describing the 3 dimensional position of objects
# compared to the array ground level in relation to the nomial centre of
# the array. Typically this frame will be used for describing the position
# on telescopes and equipment
#
# **Typical usage**: positions of telescopes on the ground (x, y, z)
#

source.subarray.peek()


######################################################################
# In case a layout is selected, the following line will produce a
# different output from the picture above.
#

source.subarray.select_subarray(layout, name="Prod3b layout").peek()


######################################################################
# .. figure:: ground_frame.png
#    :alt: Ground Frame
#
#    Ground Frame


######################################################################
# In this image all the telescope from the ``gamma_test.simtel.gz`` file
# are plotted as spheres in the GroundFrame.
#


######################################################################
# TiltedGroundFrame
# -----------------
#


######################################################################
# Tilted ground coordinate frame.
#
# The tilted ground coordinate frame is a cartesian system describing the
# 2 dimensional projected positions of objects in a tilted plane described
# by pointing_direction. The plane is rotated along the z_axis by the
# azimuth of the ``pointing_direction`` and then it is inclined with an
# angle equal to the zenith angle of the ``pointing_direction``.
#
# This frame is used for the reconstruction of the shower core position.
#


######################################################################
# .. figure:: tilted_ground_frame.png
#    :alt: Tilted Ground Frame
#
#    Tilted Ground Frame


######################################################################
# This image picture both the telescopes in the GroundFrame (red) and in
# the TiltedGroundFrame (green) are displayed: in this case since the
# azimuth of the ``pointing_direction`` is 0 degrees, then the plane is
# just tilted according to the zenith angle.
#
# For playing with these and with more 3D models of the telescopes
# themselves, have a look at the
# `CREED_VTK <https://github.com/thomasgas/CREED_VTK>`__ library.
#
