.. _calib:

.. temporary worakaround to at least have calib in the title,
   the reason is the usage of autosummary in the Reference/API
   section below

========================================
Calibration (``calib``)
========================================

.. currentmodule:: ctapipe.calib


Introduction
============

This module include all the functions and classes needed for the Calibration of CTA data.

It consists in four main sub-modules:

* :ref:`calib_camera`

* Array Calibration

* Atmosphere Calibration

* Pointing Calibration

For more information on where you should implement your code, please have a look to the README.rst files inside each directory.


Getting Started
===============

TODO: add examples.

Submodules
==========

.. toctree::
  :maxdepth: 1
  :glob:

  index_*



Reference/API
=============

.. What follows is a *temporary* workaround to circumvent
   various warnings of duplicate references caused by
   calling automodapi on the camera package.

ctapipe.calib Package
---------------------

Calibration

Classes
^^^^^^^

.. autosummary::
    ~camera.CameraCalibrator
    ~camera.GainSelector
