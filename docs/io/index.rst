.. _io:

====================
 Input/Output (`io`)
====================

.. currentmodule:: ctapipe.io

Introduction
============

`ctapipe.io` contains functions and classes related to reading data, including
camera event data.

The primary aspects for reading event data with `ctapipe` are:

* Containers - Where the event information is stored
* EventFileReaders - High level class used to read from the file
* hessio_event_source - Low level method to handle the filling of the
  containers


Container Classes
=================

Event data that is intended to be read or written from files is stored
in subclasses of `ctapipe.core.Container`, the structre of which is
defined in the `ctapipe.io.containers` module (See reference API below). Each
element in the container is a `ctapipe.core.Field`, containing the
default value, a description, and default unit if necessary. The
following rules should be followed when creating a `ctapipe.core.Container`
for new data:

* Containers both provide a way to exchange data (in-memory) between
  parts of a code, as well as define the schema for any file output
  files to be written.
* All items in a Container should be expected to be *updated at the
  same frequency*. Think of a Container as the column definitions of a
  table, therefore representing a single row in a table. For example,
  if the container has event-by-event info, it should not have an
  item in it that does not change between events (that should be in
  another container), otherwise it will be written out for each event
  and will waste space.
* a Container should be filled in all at once, not at different times
  during the data processing (to allow for parallelization and to
  avoid difficulty in reading code).
* Containers may contain a dictionary of metadata (in their `meta`
  dictionary), that will become headers in any output file (this data
  must not change per-event, etc)
* Algorithms should not update values in a container that have already
  been filled in by another algorithm. Instead, prefer a new data
  item, or a second copy of the Container with the updated values.
* Fields in a container should be one of the following:
  
 * scalar values (`int`, `float`, `bool`)
 * `numpy.NDarray` if the data are not scalar (use only simple dtypes that can
   be written to output files)
 * a `ctapipe.core.Container` class (in the case a hierarchy is needed)
 * a `ctapipe.core.Map` of `ctapipe.core.Container` or scalar values,
   if the hierarchy needs multiple copies of the same `Container`,
   organized by some variable-length index (e.g. by `tel_id` or
   algorithm name)
   
* Fields that should *not* be in a container class:
  
 * `dicts`
 * classes that are not a subclass of `ctapipe.core.Container`
 * any other type that cannot be translated automatically into the
   column of an output table.


EventFileReader Classes
=======================

These classes provide a common high-level interface for reading data from
different sources, allowing scripts to be compatible with all event data
formats.

By using an `ctapipe.io.eventfilereader.EventFileReaderFactory` inside your
python script to read the data, the correct
`ctapipe.io.eventfilereader.EventFileReader` will be
selected, therefore making the script not care what data source is given to it.

An example of how to use `ctapipe.io.eventfilereader.EventFileReaderFactory`
inside a `ctapipe.core.tool.Tool` can be found in
examples/calibration_pipeline.py:

>>> reader = EventFileReaderFactory.produce(config=self.config, tool=self)

If a hessio file was supplied to
`ctapipe.io.eventfilereader.EventFileReaderFactory`, then reader would
be of type `ctapipe.io.eventfilereader.HessioFileReader`.


Low-Level Event Readers
=======================

Each `ctapipe.io.eventfilereader.EventFileReader` utilises a different
low-level method to correctly fill the `ctapipe.core.Container` for this
particular event data file format.

These methods provide python generators that loop through a file.

Some examples of the low-level reader methods are:

* `hessio.hessio_event_source`: provides a convenient wrapper to
  reading *simtelarray* data files, like those used in CTA monte-carlo
  productions. It requires the `pyhessio` package to be installed (see
  :ref:`getting_started` for instructions installing `pyhessio`).
  This low-level method is called by
  `ctapipe.io.eventfilereader.HessioFileReader`.
* `toymodel.toymodel_event_source`: generates toy-monte-carlo dummy images for
  testing purposes
* `zfits.zfits_event_source`: reads zfits raw event files

.. figure:: shower.png
	    
   an image read from a *simtelarray* data file.


Unofficial-CTA/Prototype Camera Readers
=======================================

As the official data format for CTA is still undefined, individual camera
protoptype teams have created their own data formats in which
their camera events are stored.

Details about these readers, and the guidelines to follow to create your own,
can be found in :ref:`io_unofficial`.

Serialization of Containers:
============================

The `serializer` module provide support for storing
`ctapipe.io.Container` classes in output files (for example FITS
tables or pickle files)

The `hdftableio` submodule provides an API to write/read Containers to and
from HDF5 tables using the pytables package.


Submodules
==========

.. toctree::
  :maxdepth: 1
  :glob:

  index_*


Reference/API
=============

.. automodapi:: ctapipe.io.eventfilereader

------------------------------
       
.. automodapi:: ctapipe.io.hessio

------------------------------
		
.. automodapi:: ctapipe.io.containers
    :no-inheritance-diagram:

------------------------------

.. automodapi:: ctapipe.io.serializer
    :no-inheritance-diagram:

------------------------------

.. automodapi:: ctapipe.io.hdftableio

