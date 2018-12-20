.. _coordinates:

==============================
Coordinates (`coordinates`)
==============================

.. currentmodule:: ctapipe.coordinates

Introduction
============

`ctapipe.coordinates` contains coordinate frame definitions and
coordinate transformation routines that are associated with the
reconstruction of Cherenkov telescope events.  It is built on
`astropy.coordinates`, which internally use the ERFA coordinate
transformation library, the open-source-licensed fork of the IAU SOFA
system.

Maybe we'll use the `gwcs <http://gwcs.readthedocs.org/en/latest/>`__ package
for generalised world coordinate transformations to implement some coordinate
transformations, e.g. precision pointing corrections using a pointing model.

.. note::

  Due to the underlying implementation of astropy coordinates and time
  representations, the transformation of a single coordinate between
  frames or time systems has a small overhead in speed.
  This can be mitigated by transforming many coordinates at once, which is
  much faster. All routines can accept arrays instead of single values.



Getting Started
===============

The coordinate library defines a set of *frames*, which represent
different coordinate reprentations. Coordinates should be described
using an `astropy.coordinates.SkyCoord` in the appropriate frame.

The following special frames are defined for CTA:

* `CameraFrame`
* `GroundFrame`
* `TiltedGroundFrame`
* `NominalFrame`
* `HorizonFrame`
* `TelescopeFrame`

they can be transformed to and from any other `astropy.coordinates` frame, like
`astropy.coordinates.ICRS` (RA/Dec)


        
Reference/API
=============

.. automodapi:: ctapipe.coordinates
    :no-inheritance-diagram:
