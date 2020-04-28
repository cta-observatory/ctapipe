.. _calib_camera:

##################
Camera Calibration
##################

.. currentmodule:: ctapipe.calib.camera


************
Introduction
************

This module contains all the methods and classes that are used to calibrate the
CTA Cameras (MC, prototypes and final camera calibration algorithms).


****************
CameraCalibrator
****************

The primary class in this module is the `CameraCalibrator`. This class handles
two data level transition stages for the event:

* R1 -> DL0 (:ref:`image_reducers`)
* DL0 -> DL1 (:ref:`image_charge_extractors`)

The class takes a ctapipe event container, and fills the
next data level containers with the calibrated information.

See the `CTA High-Level Data Model Definitions SYS-QA/160517
<https://jama.cta-observatory.org/perspective.req?projectId=6&docId=26528>`_ document (CTA internal) for information about the
different data levels.


*************
Reference/API
*************

.. automodapi:: ctapipe.calib.camera

------------------------------

.. automodapi:: ctapipe.calib.camera.calibrator
    :no-inheritance-diagram:

