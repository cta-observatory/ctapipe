.. _coordinates:

************************************
Coordinates (`~ctapipe.coordinates`)
************************************

.. currentmodule:: ctapipe.coordinates


Introduction
============

`ctapipe.coordinates` contains coordinate frame definitions and
coordinate transformation routines that are associated with the
reconstruction of Cherenkov telescope events.
It is built on ``astropy.coordinates``,
which internally use the ERFA coordinate transformation library,
the open-source-licensed fork of the IAU SOFA system.


Getting Started
===============

The coordinate library defines a set of *frames*,
which represent different coordinate representations.
Coordinates should be described using an ``astropy.coordinates.SkyCoord`` in the appropriate frame.

The following special frames are defined for CTA:

* `CameraFrame`
* `EngineeringCameraFrame`
* `TelescopeFrame`
* `NominalFrame`
* `GroundFrame`
* `TiltedGroundFrame`

they can be transformed to and from any other `astropy.coordinates` frame, like
`astropy.coordinates.AltAz` or `astropy.coordinates.ICRS` (RA/Dec)

The three different coordinate frames are shown here:

.. plot:: api-reference/coordinates/plot_camera_frames.py


The `CameraFrame` is used internally in ``ctapipe`` and comes from ``sim_telarray``.
It abstracts the transformation differences between 1 and 2 mirror telescopes away.
The `EngineeringCameraFrame` is used by MAGIC, FACT and the H.E.S.S. analysis
software. Finally the `TelescopeFrame` shows the camera in angular coordinates on the sky, centered on the observation position for a given telescope.


Reference/API
=============

.. automodapi:: ctapipe.coordinates
    :no-inheritance-diagram:

.. automodapi:: ctapipe.coordinates.camera_frame
    :no-inheritance-diagram:

.. automodapi:: ctapipe.coordinates.telescope_frame
    :no-inheritance-diagram:

.. automodapi:: ctapipe.coordinates.nominal_frame
    :no-inheritance-diagram:

.. automodapi:: ctapipe.coordinates.ground_frames
    :no-inheritance-diagram:
