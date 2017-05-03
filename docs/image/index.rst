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
* a description of the Camera geometry (pixel positions, etc), usually a `CameraGeometry` object


This module contains the following sub-modules, but the most important functions of each are imported into the `ctapipe.image` namespace

* `cleaning` : image noise suppression
* `hillas`: image moment parameterization
* `toymodel`: fake shower image generation for testing purposes
* `pixel_likelihood`: generates the likelihood of a pixel intensity, given an expectation value
* `charge_extractors`: extracts charge from the waveform, resulting in a single number per pixel
* `waveform_cleaners`: cleans the waveform, e.g. applying filters, convolutions, or baseline subtractions
* `reductors`: performs data volume reduction
  
Reference/API
=============


.. automodapi:: ctapipe.image
    :no-inheritance-diagram:

------------------------------
    
.. automodapi:: ctapipe.image.toymodel
    :no-inheritance-diagram:
  
.. plot:: image/image_example.py
    :include-source:


------------------------------


.. automodapi:: ctapipe.image.hillas

------------------------------
		
.. automodapi:: ctapipe.image.cleaning

An example of image cleaning and dilation:
		
.. image:: dilate.png

------------------------------
	   
.. automodapi:: ctapipe.image.pixel_likelihood

------------------------------

.. automodapi:: ctapipe.image.charge_extractors

------------------------------

.. automodapi:: ctapipe.image.waveform_cleaning

------------------------------

.. automodapi:: ctapipe.image.reductors

