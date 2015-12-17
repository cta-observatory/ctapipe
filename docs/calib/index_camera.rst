.. _calib_camera:

============
 Camera calibration
============

.. currentmodule:: ctapipe.calib.camera

This module contains the functions to calibrate the CTA cameras.

These functions are divided into the different python modules:

* MC Camera calibration (`mc.py`)

.. plot:: ../ctapipe/calib/camera/mc.py
    :caption: my_mc.py
    :name: mc.py

* Camera prototypes:

    * LST Camera (`lstcam.py`)
    * MST-NectarCam Camera (`nectarcam.py`)
    * MST-FlashCam Camera (`flashcam.py`)
    * SST-GCT Camera (`gctcam.py`)
    * SST-ASTRI Camera (`astricam.py`) 
    * SST-1M Camera (`digicam.py`) 
    * MST-SCT Camera (`sctcam.py`) 

* Camera Calibration (`camera_calibration.py`)

For a more detailed description of each module and where the different developers should include their code, see README.rst in ctapipe/calib/camera/ directory.


Reference/API
=============

.. automodapi:: ctapipe.calib.camera
    :no-inheritance-diagram:
