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
* `muon`: muon detection and parameterization
* `hillas`: image moment parameterization

.. plot:: image/image_example.py
    :include-source:

Reference/API
=============

.. automodapi:: ctapipe.image
    :no-inheritance-diagram:

.. automodapi:: ctapipe.image.mock
    :no-inheritance-diagram:
