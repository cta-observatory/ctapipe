.. _calib_array:

============
 Array calibration
============

.. currentmodule:: ctapipe.calib.array

This module contains the functions to cross- and inter-calibrate the CTA telescopes.

These functions are divided into the different python modules, corresponding to the diffferent calibration tasks defined for array calibration:

* *Array calibration using CTA data*:

    * Muon (`muon.py`)
    * Air showers (`cr.py`)
    * Cherenkov Transparency Coefficient (`ctc_tel.py`)
    * Electrons (`electrons.py`)

* *Array calibration using non-CTA data*:

    * Space detectors in operation (`space_detectors_calib.py`)
    * Archival data (`archival_data_calib.py`)
