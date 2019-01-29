.. _image:

=================
Imaging (`image`)
=================

.. currentmodule:: ctapipe.image

`ctapipe.image` contains all algortihms that operate on Cherenkov camera images.

A *Cherenkov image* is defined as two pieces of data:

* a `numpy` array of pixel values (which can either be 1D, or 2D if time samples are included)
* a description of the Camera geometry (pixel positions, etc), usually a `ctapipe.instrument.CameraGeometry` object


This module contains the following sub-modules, but the most important functions of each are imported into the `ctapipe.image` namespace

.. toctree::
  :maxdepth: 1
  :glob:

  hillas
  cleaning
  muon
  toymodel
  pixel_likelihood
  charge_extractors
  reducers
  geometry_converter



Reference/API
=============


.. automodapi:: ctapipe.image
    :no-inheritance-diagram:
