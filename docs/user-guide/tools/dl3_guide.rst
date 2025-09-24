.. _dl3_guide:

***************************************************************
How to produce DL3 for observations using ``ctapipe`` tools
***************************************************************

The guide explains how to obtain a DL3 file (gamma-like events with IRFs) for your observations.

.. note::
   * This guide assumes you have Monte Carlo simulations processed to obtain RF and IRFs. For more details, see :doc:`IRF guide <irf_guide>` and :doc:`DL2 guide <dl2_guide>`.
   * Use the same ctapipe configuration files for processing MC and observations in all steps, ensuring the same processing is applied to both.

Setup
=====

First define the following environment variables:

* ``DL0_FOLDER`` — directory with the DL0 data
* ``DL1_FOLDER`` — directory with the DL1 data
* ``DL2_FOLDER`` — directory with the DL2 data
* ``DL3_FOLDER`` — directory with the DL3 data
* ``RF_FOLDER`` — directory with the random forest models for the reconstruction
* ``IRF_FOLDER`` — directory with the IRFs and associated cuts file
* ``CONFIG_DL1`` — configuration file for the DL0→DL1 processing, e.g. ``optimize_dl0_to_d1.yaml``
* ``CONFIG_DL3`` — configuration file for the DL2→DL3 processing of observations
  (quality cuts must be identical to those used for optimized cuts and IRFs),
  e.g. ``optimize_dl2_to_d3_obs.yaml``

Running the tools
=================

1) DL0 → DL1
------------

Process your DL0 file to DL1 level:

.. code-block:: bash

   ctapipe-process \
     --input "$DL0_FOLDER/MyRun_subrun_xxx.dl0.h5" \
     --output "$DL1_FOLDER/MyRun_subrun_xxx.dl1.h5" \
     --config "$CONFIG_DL1" \
     --provenance-log "$DL1_FOLDER/MyRun_subrun_xxx.dl1.provenance.log" \
     --log-file "$DL1_FOLDER/MyRun_subrun_xxx.dl1.log"

If your observation is divided into subruns, merge them:

.. code-block:: bash

   ctapipe-merge --progress \
     --telescope-events \
     --dl1-parameters \
     --no-dl1-images \
     --single-ob \
     -i "$DL1_FOLDER" \
     -o "$DL1_FOLDER/MyRun.dl1.h5" \
     -l "$DL1_FOLDER/MyRun_subrun_xxx.dl1.log"

2) DL1 → DL2 (apply RF models)
------------------------------

Apply the RF models trained on MC to reconstruct events and obtain the DL2 file:

.. code-block:: bash

   ctapipe-apply-models \
     --input "$DL1_FOLDER/MyRun.dl1.h5" \
     --output "$DL2_FOLDER/MyRun.dl2.h5" \
     --reconstructor "$RF_FOLDER/energy_regressor.pkl" \
     --reconstructor "$RF_FOLDER/particle_classifier.pkl" \
     --reconstructor "$RF_FOLDER/disp_reconstructor.pkl" \
     --provenance-log "$DL2_FOLDER/MyRun.provenance.log" \
     --log-file "$DL2_FOLDER/MyRun.dl2.log" \
     --log-level INFO

.. note::
   The option ``--reconstructor "$RF_FOLDER/particle_classifier.pkl"`` is only required for **monoscopic** reconstructions.

3) DL2 → DL3
------------

You could finally produce your DL3 file:

.. code-block:: bash

   ctapipe-create-dl3 \
     --dl2-file "$DL2_FOLDER/MyRun.dl2.h5" \
     --output "$DL3_FOLDER/MyRun.dl3.fits.gz" \
     --irfs-file "$IRF_FOLDER/MyIRFs.fits" \
     --cuts "$IRF_FOLDER/MyCuts.fits" \
     -c "$CONFIG_DL3" \
     --no-optional-columns \
     --provenance-log "$DL3_FOLDER/MyRun.dl3.provenance.log" \
     --log-file "$DL3_FOLDER/MyRun.dl3.log" \
     --log-level INFO

The DL3 file is now ready to be used for high level analysis.
