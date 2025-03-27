.. _irf_guide:

***********************************************************************************
How to use the cut optimization and the irf tool to create an IRF using ``ctapipe``
***********************************************************************************

This guide explains how to create an IRF using two ``ctapipe`` tools from DL2 files.
The first step is the optimization of gamma/hadron cuts and cuts on the origin direction
using `ctapipe-optimize-event-selection <ctapipe.tools.optimize_event_selection.EventSelectionOptimizer>`.
The second step is the calculation of the IRF itself using `ctapipe-compute-irf <ctapipe.tools.compute_irf.IrfTool>`,
which also applies the previously optimized g/h cuts and optionally the cuts on the origin direction.

.. NOTE::
   * This guide assumes you have a directory containing gamma, proton, and electron files
     already containing reconstructed energy, particle type, and direction (DL2).
   * The provided commands also assume you are trying to process files in a ``bash`` shell
     environment.

Setup
=====
First define the following environment variables. It is recommended to use different files
for the cut optimization and the calculation of the IRF to avoid overestimating the performance
due to the cuts being over-adjusted for the individual files.

* ``OUTPUT_DIR``, the directory in which to save all the output files
* ``OPT_CONFIG_FILE``, a configuration file for the optimization step, e.g. ``optimize_cuts.yaml``
* ``IRF_CONFIG_FILE``, a configuration file for making irfs without a cut on the origin direction, e.g. ``compute_irf.yaml``
* ``GAMMA_OPT_FILE``, the gamma file used to optimize selection cuts
* ``PROTON_OPT_FILE``, the proton file used to optimize selection cuts
* ``ELECTRON_OPT_FILE``, the electron file used to optimize selection cuts
* ``GAMMA_IRF_FILE``, the gamma file used to derive the final instrument response
* ``PROTON_IRF_FILE``, the proton file used to derive the final instrument response
* ``ELECTRON_IRF_FILE``, the electron file used to derive the final instrument response

Running the tools
=================
The optimization of both, gamma/hadron cuts and cuts on the origin direction, can then be done as follows::

  ctapipe-optimize-event-selection --config $OPT_CONFIG_FILE \
    --gamma-file $GAMMA_OPT_FILE \
    --proton-file $PROTON_OPT_FILE \
    --electron-file $ELECTRON_OPT_FILE \
    --output $OUTPUT_DIR/cuts_opt.fits

After that, the IRF can be calculated while applying both sets of cuts
by passing the ``--spatial-selection-applied`` flag::

  ctapipe-compute-irf --config $IRF_CONFIG_FILE \
    --cuts $OUTPUT_DIR/cuts_opt.fits \
    --gamma-file $GAMMA_IRF_FILE \
    --proton-file $PROTON_IRF_FILE \
    --electron-file $ELECTRON_IRF_FILE \
    --output $OUTPUT_DIR/irf_spatial_cuts.fits.gz \
    --benchmark-output $OUTPUT_DIR/benchmarks_spatial_cuts.fits.gz \
    --spatial-selection-applied

Or while only applying the gamma/hadron cuts, which is the default behaviour::

  ctapipe-compute-irf --config $IRF_CONFIG_FILE \
    --cuts $OUTPUT_DIR/cuts_opt.fits \
    --gamma-file $GAMMA_IRF_FILE \
    --proton-file $PROTON_IRF_FILE \
    --electron-file $ELECTRON_IRF_FILE \
    --output $OUTPUT_DIR/irf.fits.gz \
    --benchmark-output $OUTPUT_DIR/benchmarks.fits.gz

.. NOTE::
  * If the background should not be estimated using the given simulation files, the ``--no-do-background`` flag
    can be passed. By default, the background estimation will be included.
  * If ``--benchmark-output`` is not given, the benchmarks will not be calculated.
