.. _command_line_tools:

******************
Command-line Tools
******************

You can get a list of all available command-line tools by typing

.. code-block:: sh

    ctapipe-info --tools


Data Processing Tools
=====================

* ``ctapipe-quickstart``: create some default analysis configurations and a working directory
* ``ctapipe-process``: Process event data in any supported format from R0/R1/DL0 to DL1 or DL2 HDF5 files.
* ``ctapipe-apply-models``: Tool to apply machine learning models in bulk (as opposed to event by event).
* ``ctapipe-train-disp-reconstructor`` : Train the ML models for the  `ctapipe.reco.DispReconstructor` (monoscopic reconstruction)
* ``ctapipe-train-energy-regressor``:  Train the ML models for the `ctapipe.reco.EnergyRegressor` (energy estimation)
* ``ctapipe-train-particle-classifier``: Train the ML models for the  `ctapipe.reco.ParticleClassifier` (gamma-hadron separation)

File Management Tools:
======================
* ``ctapipe-merge``:   Merge multiple ctapipe HDF5 files into one
* ``ctapipe-fileinfo``:  Display information about ctapipe HDF5 output files

Other Tools
===========

* ``ctapipe-info``:  print information about your ctapipe installation and its command-line tools.
* ``ctapipe-dump-instrument``: writes instrumental info from any supported event input file, and writes them out as FITS or ECSV files for external use.
