Atmosphere Calibration modules
==========================

This directory contains all the functions aimed to provide the calibration of the atmosphere.

* Available so far:
   * Lidar: simple elastic Klett reconstruction on HESS data in an ASCII file, plot extinction
   * Atmospheric Density profile: read and plot atmospheric density, pressure and temperature data in atmoprofN.dat format (corsika in put)
   * Atmospheric Transmission table: read and plot atmospheric transmission table in a MODTRAN like format (as used in simtel input)
   * Radio sonde: read radio sonde data in fsl format, plot temperature and pressure profiles
   * Rayleigh scattering analytical calculator for a given Pressure and Temperature

* To Be Done
   * GDAS, WRF and SENES: gdas_wrf_senes.py
   * Satellite data: satellite_atmo.py
   * Cherenkov Transparency Coefficient: ctc.py - 
   * Data selection: data_selection.py
   * Data correction: data_correction.py

* @TODO
   * pylint, pep8
   * add unit test
