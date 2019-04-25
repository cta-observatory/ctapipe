.. _calib_camera:

==================
Camera calibration
==================

.. currentmodule:: ctapipe.calib.camera

Introduction
============

This directory contains all the functions that are used to calibrate the CTA
Cameras (MC, prototypes and final camera calibration algorithms).

The calibration is seperated between data level transitions:
    * R0 -> R1 (Camera precalibration) (r1.py)
    * R1 -> DL0 (Data volume reduction) (dl0.py)
    * DL0 -> DL1 (Waveform cleaning and charge extraction) (dl1.py)

The routines in these scripts take a ctapipe event container, and fill the
next data level container with the calibrated information.
If the container is empty for the source data level then that calibration is
skipped and the target data level container is unchanged (for example
performing the calibration in r1.py on data read into the R1 container will
leave the container unchanged, as there is no data in the R0 container)

See the `CTA High-Level Data Model Definitions SYS-QA/160517
<https://jama.cta-observatory.org/perspective.req?projectId=6&docId=26528>`_ document (CTA internal) for information about the
different data levels.

Reference/API
=============

.. automodapi:: ctapipe.calib.camera

------------------------------

.. automodapi:: ctapipe.calib.camera.dl0
    :no-inheritance-diagram:

------------------------------

.. automodapi:: ctapipe.calib.camera.dl1
    :no-inheritance-diagram:

------------------------------

.. automodapi:: ctapipe.calib.camera.calibrator
    :no-inheritance-diagram:

