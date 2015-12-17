.. _calib_atmo:

============
 Atmosphere calibration
============

.. currentmodule:: ctapipe.calib.atmo

This module contains the functions to calibrate the atmosphere above CTA arrays.


These functions are divided into the different python modules, corresponding to the diffferent calibration tasks defined for atmosphere calibration:

* *Site climatology*:

    * LIDAR (`lidar.py`)
    * GDAS, WRF and SENES (`gdas_wrf_senes.py`)
    * Satellite data (`satellite_atmo.py`)
    * Cherenkov Transparency Coefficient (`ctc.py`)

* *Using atmospheric parameters on CTA data*:

    * Data selection (`data_selection.py`)
    * Data correction (`data_correction.py`)

 
