.. _core:

==================================================
Core Structures and Base Classes (`~ctapipe.core`)
==================================================

.. currentmodule:: ctapipe.core

Introduction
============

The `ctapipe.core` module contains base classes the provide developers
with the core functionality to implement an application that processes
data.

`~ctapipe.core.Container` provides a common data class,
`~ctapipe.core.Component` lets one define a module (worker, maker,
etc.) for a particular algorithm along with its user-editable
configuration parameters, and `~ctapipe.core.Tool` defines a
command-line application, complete with configuration file or
command-line parameter processing, and logging setup.  In the future
this will also handle provenance metadata.

All *ctapipe* applications should derive from these classes in order
to provide a common interface and to make use of the configuration system.

For details about creating command-line tools, see :ref:`tools`

The following shows the conceptual difference between `Tools <Tool>`_ and
`Components <Compenent>`_ with overall pipelines and stages.
Serialization and Deserialization are simply specialized Components that perform data access.

.. image:: tool-component.png


.. toctree::
  :maxdepth: 1

  traits


Reference/API
=============

.. automodapi:: ctapipe.core
    :no-inheritance-diagram:

.. automodapi:: ctapipe.core.component
    :no-inheritance-diagram:

.. automodapi:: ctapipe.core.tool
    :no-inheritance-diagram:

.. automodapi:: ctapipe.core.container
    :no-inheritance-diagram:
