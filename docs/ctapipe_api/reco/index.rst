.. _reco:

================================
Reconstruction (`~ctapipe.reco`)
================================

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

Moment-Based Stereo Reconstruction
----------------------------------

Moment-base reconstruction uses the moments of each shower image (the
*Hillas Parameters* to estimate the shower axis for each camera, and
combines them geometrically to estimate the true shower direction.

The implementation is in the `~ctapipe.reco.HillasReconstructor` class.

Template-Based Stereo Reconstruction
------------------------------------

Moment-base reconstruction uses the a fit of the full camera images to an expected
image model to find the best fit shower axis, energy and depth of maximum.
The implementation is in the `~ctapipe.reco.ImPACTReconstructor` class.


.. toctree::
    ImPACT

Reference/API
=============

.. automodapi:: ctapipe.reco
    :no-inheritance-diagram:
.. automodapi:: ctapipe.reco.hillas_intersection
    :no-inheritance-diagram:
.. automodapi:: ctapipe.reco.hillas_reconstructor
    :no-inheritance-diagram:
.. automodapi:: ctapipe.reco.impact
    :no-inheritance-diagram:
.. automodapi:: ctapipe.reco.reco_algorithms
    :no-inheritance-diagram:
