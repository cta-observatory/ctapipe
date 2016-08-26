.. _instrument:

===========
 Instrument
===========

.. currentmodule:: ctapipe.instrument

Introduction
============

The `ctapipe.instrument` module contains classes and methods for
describing the instrumental layout and configuration.


Getting Started
===============

Relationship between different configuration objects.  The arrows show
what information is needed to look up the next level of configuration.
In **blue** are items that change for every *run*, in **green** for every
*run-type*, and in **white** for every *array version* (which is updated
when there are changes to the array, telescope, or camera
configurations associated with a site)

.. graphviz:: config.dot


Reference/API
=============

.. automodapi:: ctapipe.instrument.obsconfig


