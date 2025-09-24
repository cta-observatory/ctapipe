.. _dl3_guide:

***************************************************************
How to produce DL3 for observations using ``ctapipe`` tools
***************************************************************

The guide explains how to obtains DL3 file (gamma-like events with IRFs) for your observations.

.. NOTE::
    * This guide assumes you have your monte carlo simulation processed to obtained RF and IRFs, for more help on how to perform the processing of monte carlo see :doc: `irf_guide` and :doc: `dl2_guide`
    * You should use for all the steps the same ctapipe configuration files for processing MC and observations, therefore ensuring than the same processing is applied to MC and observations


Setup
=====
First define the following environment variables.

* ``DL0_FOLDER``, directory with the DL0 data
* ``DL1_FOLDER``, directory with the DL1 data
* ``DL2_FOLDER``, directory with the DL2 data
* ``DL3_FOLDER``, directory with the DL3 data
* ``RF_FOLDER``, directory with the random forest for the reconstruction
* ``IRF_FOLDER``, directory with the IRFs and associated cuts file
* ``CONFIG_DL1``, a configuration file for the DL0 to DL1 processing, e.g. ``optimize_dl0_to_d1.yaml``
* ``CONFIG_DL3``, a configuration file for the DL2 to DL3 processing of observations (quality cuts need to be identical to the ones for optimsed cuts and IRFs), e.g. ``optimize_dl2_to_d3_obs.yaml``

Running the tools
=================

The first steps is to processed your DL0 file to DL1 level :

```
ctapipe-process --input $DL0_FOLDER/MyRun_subrun_xxx.dl0.h5
                --output $DL1_FOLDER/MyRun_subrun_xxx.dl1.h5
                --config $CONFIG_DL1
                --provenance-log $DL1_FOLDER/MyRun_subrun_xxx.dl1.provenance.log
                --log-file $DL1_FOLDER/MyRun_subrun_xxx.dl1.log
```

If your observations is divided in subruns, you could then merge them :

```
ctapipe-merge --progress
              --telescope-events
              --dl1-parameters
              --no-dl1-images
              --single-ob
              -i $DL1_FOLDER
              -o $DL1_FOLDER/MyRun.dl1.h5
              -l $DL1_FOLDER/MyRun_subrun_xxx.dl1.log
```

Then the RF could be applied to perform the reconstruction and obtained DL2 file :

```
ctapipe-apply-models --input $DL1_FOLDER/MyRun.dl1.h5 \
                     --output $DL2_FOLDER/MyRun.dl2.h5 \
                     --reconstructor $RF_FOLDER/energy_regressor.pkl \
                     --reconstructor $RF_FOLDER/particle_classifier.pkl \
                     --reconstructor $RF_FOLDER/disp_reconstructor.pkl \
                     --provenance-log $DL2_FOLDER/MyRun.provenance.log \
                     --log-file $DL2_FOLDER/MyRun.dl2.log \
                     --log-level INFO
```

To note that the option `--reconstructor RF_FOLDER/particle_classifier.pkl` is only required for monocopic reconstructions.

Finnaly you could produce your DL3 file :

```
ctapipe-create-dl3 --dl2-file $DL2_FOLDER/MyRun.dl2.h5 \
                   --output $DL3_FOLDER/MyRun.dl3.fits.gz \
                   --irfs-file $IRF_FOLDER/MyIRFs.fits \
                   --cuts $IRF_FOLDER/MyCuts.fits \
                   -c $CONFIG_DL3 \
                   --no-optional-columns \
                   --provenance-log $DL3_FOLDER/MyRun.dl3.provenance.log \
                   --log-file $DL3_FOLDER/MyRun.dl3.log \
                   --log-level INFO"""
```
