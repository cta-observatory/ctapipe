.. _coordinates:

===========
Coordinates
===========

.. currentmodule:: ctapipe.coordinates

Introduction
============

`ctapipe.coordinates` contains coordinate frame definitions and coordinate
transformation routines.
It is built on `astropy.coordinates`.

Maybe we'll use the `gwcs <http://gwcs.readthedocs.org/en/latest/>`__ package
for generalised world coordinate transformations to implement some coordinate
transformations, e.g. precision pointing corrections using a pointing model.

Getting Started
===============

Coordinate frames and transformations aren't implemented yet.

Except for this super-simple example of dividing by the focal length ...

.. code-block:: python

    >>> import astropy.units as u
    >>> from ctapipe.coordinates import CameraFrame, TelescopeFrame
    >>> camera_coord = CameraFrame(x=1*u.m, y=2*u.m, z=0*u.m)
    >>> print(camera_coord)
    <CameraFrame Coordinate: (x, y, z) in m
        (1.0, 2.0, 0.0)>
    >>> telescope_coord = camera_coord.transform_to(TelescopeFrame)
    >>> print(telescope_coord)
    <TelescopeFrame Coordinate (focal_length=15.0 m): (x, y, z) [dimensionless]
        (0.06666667, 0.13333333, 0.0)>

Reference/API
=============

.. automodapi:: ctapipe.coordinates
    :no-inheritance-diagram:
