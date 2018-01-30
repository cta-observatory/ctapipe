.. _io:

====================
 Input/Output (`io`)
====================

.. currentmodule:: ctapipe.io

Introduction
============

`ctapipe.io` contains functions and classes related to reading, writing, and
in-memory storage of event data


Reading Event Data
===================

This module provides a set of *event sources* that are python
generators that loop through an input file or stream and fill in
`ctapipe.core.Container` classes, defined below. They are designed such that
ctapipe can be independent of the file format used for event data, and new
formats may be supported by simply adding a plug-in.

The underlying mechanism is a set of `EventSource` sub-classes that
read data in various formats, with a common interface and automatic command-line
configuration parameters. These are generally constructed in a generic way by
using `EventSourceFactory.produce` or by using the helper function
`event_source(file_or_url)`, both of which will construct the
appropriate `EventSource` subclass based on the input file's type.
The former is recommended when configuration information should be
passed to a `ctapipe.core.Tool`, while the latter helper function qis useful
for small scripts or interactive use.

The resulting `EventSource`  then works like a python collection and can be
looped over, providing data for each subsequent event. If looped over
multiple times, each will start at the beginning of the file (except in
the case of streams that cannot be restarted):

.. code-block:: python3

  with EventSourceFactory.produce("file.simtel.gz") as source:
      for event in source:
         do_something_with_event(event)


If you need random access to events rather than looping over all events in
order, you can use the `EventSeeker` class to allow random access by *event
index* or *event_id*. This may not be efficient for some `EventSources` if
the underlying file type does not support random access.


Container Classes
=================

Event data that is intended to be read or written from files is stored
in subclasses of `ctapipe.core.Container`, the structre of which is
defined in the `containers` module (See reference API below). Each
element in the container is a `ctapipe.core.Field`, containing the
default value, a description, and default unit if necessary. The
following rules should be followed when creating a `Container` for new
data:

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
 * `numpy.NDarray` if the data are not scalar (use only simple dtypes that can be written to output files)
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


Serialization of Containers:
============================

The `serializer` module provide support for storing
`ctapipe.io.Container` classes in output files (for example FITS
tables or pickle files)

The `hdftableio` submodule provides an API to write/read Containers to and
from HDF5 tables using the pytables package.


Reference/API
=============

.. automodapi:: ctapipe.io

------------------------------
		
.. automodapi:: ctapipe.io.containers
    :no-inheritance-diagram:

------------------------------

.. automodapi:: ctapipe.io.serializer
    :no-inheritance-diagram:

