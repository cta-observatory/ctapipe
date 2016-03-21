Array Calibration modules
==========================

This directory contains all the functions that are used to relative calibrate the different telescopes within telescopes of identical size and between different sizes.

* Array calibration using CTA data:
  * Muon: muon.py
  * Air showers: cr.py
  * Cherenkov Transparency Coefficient: ctc_tel.py
  * Electrons: electrons.py

* Array calibration using non-CTA data:
  * Space detectors in operation: space_detectors_calib.py
  * Archival data: archival_data_calib.py

TO DEVELOPERS: please, include all the functions required for your calibration method inside your corresponding file. 
If you need to organize changes into other modules, please contact first your DATA integration responsible (see DATA tasks in Redmine).