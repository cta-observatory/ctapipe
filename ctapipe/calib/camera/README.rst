Camera Calibration modules
==========================

This directory contains all the functions that are used to calibrate the CTA Cameras (MC, prototypes and final camera calibration algorithms).

* MC (mc.py): contains the functions to calibrate MC-prod2 files. An example of how to use them can be found in ctapipe/examples/calibration_pipeline.py.

* Camera prototypes (xSTcam.py): the different camera prototypes will implement their calibration functions into their corresponding python modules:

  * template: mycam.py
  * LST Cam: lstcam.py
  * MST-NectarCam: nectarcam.py
  * MST-FlashCam: flashcam.py
  * SST-GCT Cam: gctcam.py
  * SST-ASTRI Cam: astricam.py
  * SST-1M DigiCam: digicam.py
  * MST-SCT Cam: sctcam.py

TO CAMERA PROTOTYPES DEVELOPERS: please, include all the functions used to calibrate your camera inside your corresponding file. 
In order to easy track the different calibration function we recomend to follow the template 'mycam.py'

* Camera calibration (camera_calibration.py): contains all the final camera calibration algorithms for the CTA cameras.
