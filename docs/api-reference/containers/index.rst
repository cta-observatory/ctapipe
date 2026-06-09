.. _containers:

**********************************
Containers (`~ctapipe.containers`)
**********************************

.. currentmodule:: ctapipe.containers


Introduction
============

The `ctapipe.containers` module contains the data model definition of all
ctapipe `~ctapipe.core.Container` classes, which provide the container definitions for all
ctapipe data.

The main data structure is the Container for a subarray event: `~ctapipe.containers.SubarrayEventContainer`
which contains the subarray-level information and a collection of `TelescopeEventContainers <ctapipe.containers.TelescopeEventContainer>`.


Reference/API
=============

.. automodapi:: ctapipe.containers
