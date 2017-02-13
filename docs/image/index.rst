.. _image:

=================
Imaging (`image`)
=================

.. currentmodule:: ctapipe.image

Introduction
============

`ctapipe.image` contains all algortihms that operate on Cherenkov camera images.

A *Cherenkov image* is defined as two pieces of data:

* a `numpy` array of pixel values (which can either be 1D, or 2D if time samples are included)
* a description of the Camera geometry (pixel positions, etc), usually a `io.CameraGeometry` object


This module contains the following sub-modules:

* `cleaning` : image noise suppression
* `hillas`: image moment parameterization
* `toymodel`: fake shower image generation for testing purposes
* `pixel_likelihood`: generates the likelihood of a pixel intensity, given an expectation value


Reference/API
=============

.. automodapi:: ctapipe.image
    :no-inheritance-diagram:

.. automodapi:: ctapipe.image.toymodel
    :no-inheritance-diagram:

Toymodel Example
^^^^^^^^^^^^^^^^
  
.. plot:: image/image_example.py
    :include-source:

.. automodapi:: ctapipe.image.hillas

.. automodapi:: ctapipe.image.cleaning

.. automodapi:: ctapipe.image.pixel_likelihood
