.. _reco:

********************************
Reconstruction (`~ctapipe.reco`)
********************************

.. currentmodule:: ctapipe.reco


Introduction
============

`ctapipe.reco` contains functions and classes to reconstruct physical
shower parameters, using either stereo (multiple images of a shower)
or mono (single telescope) information.

All shower reconstruction algorithms should be subclasses of
`~ctapipe.reco.Reconstructor` which defines some common functionality.


Currently Implemented Algorithms
================================


Moment-based Stereo Reconstruction
----------------------------------

Moment-base reconstruction uses the moments of each shower image (the
*Hillas Parameters* to estimate the shower axis for each camera, and
combines them geometrically to estimate the true shower direction.

The implementation is in the `~ctapipe.reco.HillasReconstructor` class.


Machine Learning-based Reconstruction
-------------------------------------

This module also provides `~ctapipe.reco.Reconstructor` implementations using
machine learning algorithms.

At the moment, these are based on algorithms from ``scikit-learn`` and
make use of DL1b and DL2 information.


Template-based Stereo Reconstruction
------------------------------------

Moment-base reconstruction uses the a fit of the full camera images to an expected
image model to find the best fit shower axis, energy and depth of maximum.
The implementation is in the `~ctapipe.reco.ImPACTReconstructor` class.


.. toctree::

  sklearn
  stereo_combination
  ImPACT


Reference/API
=============

.. automodapi:: ctapipe.reco
    :no-inheritance-diagram:

.. automodapi:: ctapipe.reco.reconstructor
    :no-inheritance-diagram:

.. automodapi:: ctapipe.reco.hillas_intersection
    :no-inheritance-diagram:

.. automodapi:: ctapipe.reco.hillas_reconstructor
    :no-inheritance-diagram:

.. automodapi:: ctapipe.reco.impact
    :no-inheritance-diagram:
