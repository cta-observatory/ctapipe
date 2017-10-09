Atmosphere Calibration modules
==========================

This directory contains all the functions aimed to provide the calibration of the atmosphere.

* Site climatology:
  * LIDAR: lidar.py
  * GDAS, WRF and SENES: gdas_wrf_senes.py
  * Satellite data: satellite_atmo.py
  * Radio sonde : read radio sonde data in fsl format - radiosonde_reader.py
  * Cherenkov Transparency Coefficient: ctc.py
  * Rayleigh scattering analytical calculator: scattering_rayleigh.py

* Application of atmospheric parameters on CTA data:
  * Data selection: data_selection.py
  * Data correction: data_correction.py

* To be implemented
  * https://github.com/bregeon/HESSLidarTools/blob/master/src/AtmoAbsorption.C
  * https://github.com/bregeon/HESSLidarTools/blob/master/src/AtmoPlotter.C
  * https://github.com/bregeon/HESSLidarTools/blob/master/src/AtmoProfile.C
  * https://github.com/bregeon/HESSLidarTools/blob/master/src/RayleighScattering.C

