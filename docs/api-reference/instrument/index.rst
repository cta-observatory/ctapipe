.. _instrument:

**********************************
Instrument (`~ctapipe.instrument`)
**********************************

.. currentmodule:: ctapipe.instrument


Introduction
============

The `~ctapipe.instrument` module contains classes and methods for
describing the instrumental layout and configuration.

This module is under heavy restructuring and should not be considered
ready for general use, except for the `CameraGeometry` object, which
provides pixel positions, etc.


Hierarchy of InstrumentDescription Classes
==========================================

* `SubarrayDescription` (describes full subarray)

  * `TelescopeDescription` (describes a single telescope)

    * `OpticsDescription` (describes the optical support structure and mirror)

    * `CameraDescription` (describes the camera)

        * `CameraGeometry` (describes the geometrical aspects of the camera, e.g.
          only that which is needed by reconstruction methods)

        * `CameraReadout` (describes the aspects related to the waveform
          readout e.g. sampling rate and reference pulse shape.geometrical
          aspects of the camera, e.g.

        * [to come: classes to hold more detailed hardware-level info about a
          camera]


.. toctree::
  :maxdepth: 1

  subarray
  telescope
  camera
  optics


Reference/API
=============

.. automodapi:: ctapipe.instrument
    :no-inheritance-diagram:
