.. _monitoring:

**********************************
Monitoring (`~ctapipe.monitoring`)
**********************************

.. currentmodule:: ctapipe.monitoring

Monitoring data are time-series used to monitor the status or quality of hardware, software algorithms, the environment, or other data products. These contain values recorded periodically at different rates, and can be thought of as a set of tables with rows identified by a time-stamp. They are potentially acquired during the day or nighttime operation of the array and during subsequent data processing, but ataverage rates much slower than Event data and faster than the length of a typical observation block. Examples include telescope tracking positions, trigger rates, camera sensor conditions, weather conditions, and the status or quality-control data of a particular hardware or software component.

This module provides some code to help to generate monitoring data from processed event data, particularly for the purposes of calibration and data quality assessment.

Code related to :ref:`stats_aggregator`, :ref:`calibration_calculator`, and :ref:`outlier_detector` is implemented here.


Submodules
==========

.. toctree::
  :maxdepth: 1

  aggregator
  calculator
  interpolation
  outlier


Reference/API
=============

.. automodapi:: ctapipe.monitoring
    :no-inheritance-diagram:
