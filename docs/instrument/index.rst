.. _instrument:

=========================
Instrument (`instrument`)
=========================

.. currentmodule:: ctapipe.instrument

Introduction
============

The `ctapipe.instrument` module contains classes and methods for
describing the instrumental layout and configuration.

This module is under heavy restructuring and should not be considered
ready for general use, except for the `CameraGeometry` object, which
provides pixel positions, etc.

Hierarchy of InstrumentDescription Classes
==========================================

* `SubarrayDescription` (describes full subarray)

  * `TelescopeDescription` (describes a single telescope)

    * `OpticsDescription` (describes the optical support structure and mirror)
    * `CameraGeometry` (describes the geometrical aspects of the camera, e.g.
      only that which is needed by reconstruction methods)
    * [to come: classes to hold more detailed hardware-level info about a
      camera]


.. toctree::
  :maxdepth: 2

  subarray
  telescope
  camera
  optics



Other Instrumental Data
=======================

Atmosphere Profiles
-------------------

With the instrument module you can also load standard atmosphere profiles,
which are read from tables located in `ctapipe_resources` by default

The function `get_atmosphere_profile_functions()` returns two inerpolation
functions that convert between height and atmosphere thickness.

Reference/API
=============

.. automodapi:: ctapipe.instrument

		


