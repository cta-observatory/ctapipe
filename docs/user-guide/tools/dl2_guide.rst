.. _dl2_guide:

***************************************************************
How to do DL2 reconstruction using ``ctapipe`` tools
***************************************************************

This guide explains how to use ``ctapipe`` to process files containing images (data level 1a/ DL1a)
to data level 2 (DL2) for both monoscopic and stereoscopic analyses.
This includes the steps of image cleaning, image parametrization, training and applying machine learning
models to estimate the energy and particle type, and either geometric direction reconstruction (stereo)
or machine learning based direction reconstruction using the ``disp`` algorithm (mono or stereo).

.. NOTE::
   * This guide assumes you have a directory containing gamma, proton, and electron files
     already processed to DL1a.
   * The provided commands assume you are trying to process files in a ``bash`` shell environment.

Setup
=====
As always, you can run

.. code-block:: sh

  ctapipe-quickstart

to get some example configuration files, which you can use as a starting point to create your
desired configuration.
You can also get up to date configuration files from `gitlab <https://gitlab.cta-observatory.org/cta-computing/dpps/datapipe/pipeline-configurations>`_,
but take care to **select the configuration files appropriate for the version of ctapipe you are using**.

To keep things organized, also define an output directory and make sure it exists, for example like this::

  OUTPUT_DIR=build
  mkdir -p $OUTPUT_DIR

Image cleaning, image parametrization, and geometric direction reconstruction
=============================================================================
Image cleaning and parametrization, as well as the geometric direction reconstruction can all be done by
`ctapipe-process <ctapipe.tools.process.ProcessorTool>` with the following command::

  ctapipe-process --input $INPUT_FILE \
    --output $OUTPUT_DIR/$OUTPUT_FILE \
    --config $CONFIG_FILE \
    --config $SUBARRAY_FILE \
    --provenance-log $OUTPUT_DIR/$PROVENANCE_LOG

``INPUT_FILE`` is the path to an input file and ``OUTPUT_FILE`` the name of the output file.
``CONFIG_FILE`` is the path to a configuration file, such as ``dl1_to_dl2.yml`` found in the gitlab
for ``v0.23/v1`` of ``ctapipe``, and ``SUBARRAY_FILE`` is another configuration file describing
the subarray, such as ``prod6/subarray_north_alpha.yml`` found next to the  ``dl1_to_dl2.yml`` in the gitlab.
You can also combine all the configuration options in one file and only pass this one to the tool.
Finally, ``PROVENANCE_LOG`` is the name of the file in which the provenance log will be saved, take care
to change it for each input file or later tracking will be complicated.

Run the above command with the inputs adjusted so that each of the available input files is processed
into a separate file, and you will end up with a set of ``h5`` files containing image parameters and
entries at the path ``dl2/event/subarray/geometry``.

To exclude the geometric direction reconstruction from the processing (e.g. for a mono analysis),
the ``ShowerProcessor`` section in ``dl1_to_dl2.yml`` would have to be changed to not include
``HillasReconstructor`` under ``reconstructor_types``.
Or, even easier, the whole ``ShowerProcessor`` section can be removed.

Merging
=======
Decide on which fraction of your files you want to use for training each ML model:

* Training the energy regressor requires a gamma training file (called e.g. ``gamma_merged_train_en.dl2.h5``).
* Training the particle classifier requires another gamma and a proton training file
  (called e.g. ``gamma_merged_train_clf.dl2.h5`` and ``proton_merged_train.dl2.h5``).
* Training the disp reconstructor requires a gamma training file, which can be the same file
  as used for the particle classifier (e.g. ``gamma_merged_train_clf.dl2.h5``).

The remaining gamma and proton files can be used for testing (or e.g. IRF calculation).
It does not make sense to use electrons for training, so all electron files can be used for testing.

Then save these sets of files into a total of six lists and run something like the following command
for each of these lists::

  ctapipe-merge $GAMMA_TRAIN_EN_FILES --output $OUTPUT_DIR/gamma_merged_train_en.dl2.h5

where ``GAMMA_TRAIN_EN_FILES`` is an environment variable containing the list of gamma files
you want to merge into a training set, for example generated using::

  GAMMA_TRAIN_EN_FILES=$(echo $INPUT_DIR/gamma*[0-1]*.h5)

which will merge all gamma files in ``INPUT_DIR`` with names that start with "0" or "1".
Alternatively you could save the files into a literal file (e.g. ``gamma_train_en_files.list``),
one file name per row, which you then use like this::

  GAMMA_TRAIN_EN_FILES=$(cat gamma_train_en_files.list)
  ctapipe-merge $GAMMA_TRAIN_EN_FILES --output $OUTPUT_DIR/gamma_merged_train_en.dl2.h5

Using some method of specifying files, merge your gamma, proton, and electron files so that you end up with six merged files:

* Gamma train energy, the file containing the gamma events to be used for training the energy regressor.
* Gamma train classifier, the file containing the gamma events to be used for training the particle classifier
  and the disp reconstructor.
* Gamma test, the file with gamma events used for "testing" the performance of the analysis.
* Proton train, the file containing the proton events to be used for training the particle classifier.
* Proton test, the file with proton events used for "testing" the performance of the analysis.
* Electron test, the file with electron events used for "testing" the performance of the analysis.

Training the machine learning models
====================================
The training process has the following steps:

1. Train an energy model on the gamma train energy file.
2. Apply the energy model to the gamma train classifier file and the proton train file.
3. Train a particle classifier on the gamma train classifier file and the proton train file.
4. Train a disp reconstructor on the gamma train classifier file.

First define the following environment variables:

* ``REG_CONF_FILE``, a configuration file for the energy regression training for example ``train_energy_regressor.yaml``
* ``CLF_CONF_FILE``, a configuration file for the particle classification training for example ``train_particle_classifier.yaml``
* ``DISP_CONF_FILE``, a configuration file for the disp reconstructor training for example ``train_disp_reconstructor.yaml``
* ``INPUT_GAMMA_EN_FILE``, the gamma train energy file created in the previous step
* ``INPUT_GAMMA_CLF_FILE``, the gamma train classifier file created in the previous step
* ``INPUT_PROTON_FILE``, the proton train file
* ``EVAL_GAMMA_FILE``, the gamma test file
* ``EVAL_PROTON_FILE``, the proton test file
* ``EVAL_ELECTRON_FILE``, the electron test file

Then the training of the machine learning models is done using the following commands::

  ctapipe-train-energy-regressor --input $INPUT_GAMMA_EN_FILE \
    --output $OUTPUT_DIR/energy_regressor.pkl \
    --config $REG_CONF_FILE \
    --cv-output $OUTPUT_DIR/cv_energy.h5 \
    --provenance-log $OUTPUT_DIR/train_energy.provenance.log \
    --log-file $OUTPUT_DIR/train_energy.log \
    --log-level INFO

  ctapipe-apply-models --input $INPUT_GAMMA_CLF_FILE \
    --output $OUTPUT_DIR/gamma_train_clf.dl2.h5 \
    --reconstructor $OUTPUT_DIR/energy_regressor.pkl \
    --provenance-log $OUTPUT_DIR/apply_gamma_train_clf.provenance.log \
    --log-file $OUTPUT_DIR/apply_gamma_train_clf.log \
    --log-level INFO

  ctapipe-apply-models --input $INPUT_PROTON_FILE  \
    --output $OUTPUT_DIR/proton_train_clf.dl2.h5 \
    --reconstructor $OUTPUT_DIR/energy_regressor.pkl \
    --provenance-log $OUTPUT_DIR/apply_proton_train.provenance.log \
    --log-file $OUTPUT_DIR/apply_proton_train.log \
    --log-level INFO

  ctapipe-train-particle-classifier --signal $OUTPUT_DIR/gamma_train_clf.dl2.h5 \
    --background $OUTPUT_DIR/proton_train_clf.dl2.h5 \
    --output $OUTPUT_DIR/particle_classifier.pkl \
    --config $CLF_CONF_FILE \
    --cv-output $OUTPUT_DIR/cv_particle.h5 \
    --provenance-log $OUTPUT_DIR/train_particle.provenance.log \
    --log-file $OUTPUT_DIR/train_particle.log \
    --log-level INFO

  ctapipe-train-disp-reconstructor --input $OUTPUT_DIR/gamma_train_clf.dl2.h5 \
    --output $OUTPUT_DIR/disp_reconstructor.pkl \
    --config $DISP_CONF_FILE \
    --cv-output $OUTPUT_DIR/cv_disp.h5 \
    --provenance-log $OUTPUT_DIR/train_disp.provenance.log \
    --log-file $OUTPUT_DIR/train_disp.log \
    --log-level INFO

which will produce three trained models saved as ``$OUTPUT_DIR/energy_regressor.pkl``, ``$OUTPUT_DIR/particle_classifier.pkl``,
and ``$OUTPUT_DIR/disp_reconstructor.pkl``.
The saved model for the disp reconstruction contains both, the regressor for estimating ``norm(disp)`` and the classifier
for determining ``sign(disp)``.

Applying the machine learning models on the test files
======================================================
Now we can apply these trained models on the test files, ``EVAL_GAMMA_FILE``, ``EVAL_PROTON_FILE``, and ``EVAL_ELECTRON_FILE``,
to produce the final DL2 files::

  ctapipe-apply-models --input $EVAL_GAMMA_FILE \
    --output $OUTPUT_DIR/gamma_final.dl2.h5 \
    --reconstructor $OUTPUT_DIR/energy_regressor.pkl \
    --reconstructor $OUTPUT_DIR/particle_classifier.pkl \
    --reconstructor $OUTPUT_DIR/disp_reconstructor.pkl \
    --provenance-log $OUTPUT_DIR/apply_gamma_final.provenance.log \
    --log-file $OUTPUT_DIR/apply_gamma_final.log \
    --log-level INFO

  ctapipe-apply-models --input $EVAL_PROTON_FILE \
    --output $OUTPUT_DIR/proton_final.dl2.h5 \
    --reconstructor $OUTPUT_DIR/energy_regressor.pkl \
    --reconstructor $OUTPUT_DIR/particle_classifier.pkl \
    --reconstructor $OUTPUT_DIR/disp_reconstructor.pkl \
    --provenance-log $OUTPUT_DIR/apply_proton_final.provenance.log \
    --log-file $OUTPUT_DIR/apply_proton_final.log \
    --log-level INFO

  ctapipe-apply-models --input $EVAL_ELECTRON_FILE \
    --output $OUTPUT_DIR/electron_final.dl2.h5 \
    --reconstructor $OUTPUT_DIR/energy_regressor.pkl \
    --reconstructor $OUTPUT_DIR/particle_classifier.pkl \
    --reconstructor $OUTPUT_DIR/disp_reconstructor.pkl \
    --provenance-log $OUTPUT_DIR/apply_electron_final.provenance.log \
    --log-file $OUTPUT_DIR/apply_electron_final.log \
    --log-level INFO

which will produce ``$OUTPUT_DIR/gamma_final.dl2.h5``, ``$OUTPUT_DIR/proton_final.dl2.h5``,
and ``$OUTPUT_DIR/electron_final.dl2.h5``.
