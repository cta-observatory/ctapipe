.. _command_line_tools:

******************
Command-line Tools
******************

You can get a list of all available command-line tools by typing

.. code-block:: sh

  ctapipe-info --tools


Data Processing Tools
=====================

* `ctapipe-quickstart <ctapipe.tools.quickstart.QuickStartTool>`: create some default analysis configurations and a working directory
* `ctapipe-process <ctapipe.tools.process.ProcessorTool>`: Process event data in any supported format from R0/R1/DL0 to DL1 or DL2 HDF5 files.
* `ctapipe-apply-models <ctapipe.tools.apply_models.ApplyModels>`: Tool to apply machine learning models in bulk (as opposed to event by event).
* `ctapipe-calculate-pixel-statistics <ctapipe.tools.calculate_pixel_stats.PixelStatisticsCalculatorTool>`: Tool to aggregate statistics
  and detect outliers from pixel-wise image data.
* `ctapipe-train-disp-reconstructor <ctapipe.tools.train_disp_reconstructor.TrainDispReconstructor>` :
  Train the ML models for the  `~ctapipe.reco.DispReconstructor` (monoscopic reconstruction)
* `ctapipe-train-energy-regressor <ctapipe.tools.train_energy_regressor.TrainEnergyRegressor>`:
  Train the ML models for the `~ctapipe.reco.EnergyRegressor` (energy estimation)
* `ctapipe-train-particle-classifier <ctapipe.tools.train_particle_classifier.TrainParticleClassifier>`:
  Train the ML models for the  `~ctapipe.reco.ParticleClassifier` (gamma-hadron separation)
* `ctapipe-optimize-event-selection <ctapipe.tools.optimize_event_selection.EventSelectionOptimizer>`:
  Calculate gamma/hadron and direction cuts (e.g. for IRF calculation).
* `ctapipe-compute-irf <ctapipe.tools.compute_irf.IrfTool>`: Calculate an IRF with or without applying a direction cut
  and optionally benchmarks.

File Management Tools:
======================
* `ctapipe-merge <ctapipe.tools.merge.MergeTool>`:   Merge multiple ctapipe HDF5 files into one
* ``ctapipe-fileinfo``:  Display information about ctapipe HDF5 output files

Other Tools
===========

* ``ctapipe-info``:  print information about your ctapipe installation and its command-line tools.
* `ctapipe-dump-instrument <ctapipe.tools.dump_instrument.DumpInstrumentTool>`: writes instrumental info from any supported event input file, and writes them out as FITS or ECSV files for external use.

Examples
========
The following pages contain examples on how to use the command-line tools.

.. toctree::
  :maxdepth: 1

  mono_dl2_guide
  irf_guide
