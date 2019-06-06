.. _coordinates:

==============================
Coordinates (`coordinates`)
==============================

.. currentmodule:: ctapipe.coordinates

Introduction
============

`ctapipe.coordinates` contains coordinate frame definitions and
coordinate transformation routines that are associated with the
reconstruction of Cherenkov telescope events.
It is built on `astropy.coordinates`,
which internally use the ERFA coordinate transformation library,
the open-source-licensed fork of the IAU SOFA system.


Getting Started
===============

The coordinate library defines a set of *frames*,
which represent different coordinate reprentations.
Coordinates should be described using an `astropy.coordinates.SkyCoord` in the appropriate frame.

The following special frames are defined for CTA:

* `CameraFrame`
* `EngineeringCameraFrame`
* `TelescopeFrame`
* `NominalFrame`
* `GroundFrame`
* `TiltedGroundFrame`

they can be transformed to and from any other `astropy.coordinates` frame, like
`astropy.coordinates.AltAz` or `astropy.coordinates.ICRS` (RA/Dec)

The two different coordinate frames are shown here:

.. plot:: ../examples/plot_camera_frames.py
    :include-source:

The `CameraFrame` is used internally and is used by H.E.S.S. and `sim_telarray`,
the `EngineeringCameraFrame` is used by `MAGIC` and `FACT`.

        
Reference/API
=============

.. automodapi:: ctapipe.coordinates
    :no-inheritance-diagram:
