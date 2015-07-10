.. _io:

==============
 Input/Output
==============

.. currentmodule:: ctapipe.reco

Introduction
============

`ctapipe.io` contains functions and classes for loading data and
support for some data structures like :class:`~ctapipe.io.camera.CameraGeometry`


Getting Started
===============


Reference/API
=============

.. automodapi:: ctapipe.io.camera
    :no-inheritance-diagram:

.. automodapi:: ctapipe.io.obsconfig


Relationship between different configuration objects.  The arrows show
what information is needed to look up the next level of configuration.
In **blue** are items that change for every *run*, in **green** for every
*run-type*, and in **white** for every *array version* (which is updated
when there are changes to the array, telescope, or camera
configurations associated with a site)

.. graphviz:: config.dot
