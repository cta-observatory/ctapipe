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
`~ctapipe.core.Container` classes, defined below. They are designed such that
ctapipe can be independent of the file format used for event data, and new
formats may be supported by simply adding a plug-in.

The underlying mechanism is a set of `~ctapipe.io.EventSource` sub-classes that
read data in various formats, with a common interface and automatic command-line
configuration parameters. These are generally constructed in a generic way by
using ``EventSource(file_or_url)`` which will construct the
appropriate `EventSource` subclass based on the input file's type.

The resulting `EventSource`  then works like a python collection and can be
looped over, providing data for each subsequent event. If looped over
multiple times, each will start at the beginning of the file (except in
the case of streams that cannot be restarted):

.. code-block:: python3

  with EventSource(input_url="file.simtel.gz") as source:
      for event in source:
         do_something_with_event(event)


If you need random access to events rather than looping over all events in
order, you can use the `EventSeeker` class to allow random access by *event
index* or *event_id*. This may not be efficient for some `EventSources <ctapipe.io.EventSource>`_ if
the underlying file type does not support random access.


Creating a New EventSource Plugin
=================================

An example can be found in:

https://github.com/cta-observatory/ctapipe_io_lst


Container Classes
=================

Event data that is intended to be read or written from files is stored
in subclasses of `~ctapipe.core.Container`, the structre of which is
defined in the `~ctapipe.containers` module (See reference API below). Each
element in the container is a `~ctapipe.core.Field`, containing the
default value, a description, and default unit if necessary. The
following rules should be followed when creating a `~ctapipe.core.Container`
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
* Containers may contain a dictionary of metadata (in their ``meta``
  dictionary), that will become headers in any output file (this data
  must not change per-event, etc)
* Algorithms should not update values in a container that have already
  been filled in by another algorithm. Instead, prefer a new data
  item, or a second copy of the Container with the updated values.
* Fields in a container should be one of the following:

 * scalar values (`int`, `float`, `bool`)
 * ``numpy.NDarray`` if the data are not scalar (use only simple dtypes that can be written to output files)
 * a `~ctapipe.core.Container` class (in the case a hierarchy is needed)
 * a `~ctapipe.core.Map` of `~ctapipe.core.Container` or scalar values,
   if the hierarchy needs multiple copies of the same `~ctapipe.core.Container`,
   organized by some variable-length index (e.g. by ``tel_id`` or
   algorithm name)

* Fields that should *not* be in a container class:

 * `dict`
 * classes that are not a subclass of `~ctapipe.core.Container`
 * any other type that cannot be translated automatically into the
   column of an output table.


Serialization of Containers:
============================

The `~ctapipe.io.TableWriter` and `~ctapipe.io.TableReader` base classes provide
an interface to implement subclasses that write/read Containers to/from
table-like data files.  Currently the only implementation is for writing
HDF5 tables via the `~ctapipe.io.HDF5TableWriter`. The output that the
`~ctapipe.io.HDF5TableWriter` produces can be read either one-row-at-a-time
using the `~ctapipe.io.HDF5TableReader`, or more generically using the
``pytables`` or ``pandas`` packages (note however any tables that have
array values in a column cannot be read into a ``pandas.DataFrame``, since it
only supports scalar values).

Writing Output Files:
=====================

The `DataWriter` Component allows one to write a series of events (stored in
`ctapipe.containers.ArrayEventContainer`) to a standardized HDF5 format file
following the data model (see :ref:`datamodels`). This includes all related datasets
such as the instrument and simulation configuration information, simulated
shower and image information, observed images and parameters and reconstruction
information. It can be used in an event loop like:

.. code-block:: python

    with DataWriter(event_source=source, output_path="events.dl1.h5") as write_data:
        for event in source:
            calibrate(event)
            write_data(event)
    
Reading Output Tables:
======================
In addition to using an `EventSource` to read R0-DL1 data files, one can also access full *tables* for files that are in HDF5 format (e.g. DL1 files).
The `read_table` function will load any table in an HDF5 table into an ``astropy.table.QTable`` in memory,
while maintaining units, column descriptions, and other ctapipe metadata.
Astropy Tables can also be converted to Pandas tables via their ``to_pandas()`` method,
as long as the table does not contain any vector columns.

.. code-block:: python

   from ctapipe.io import read_table
   mctable = read_table("events.dl1.h5", "/simulation/event/subarray/shower")
   mctable['logE'] = np.log10(mc_table['energy'])
   mctable.write("output.fits")



Standard Metadata Headers
=========================

The `ctapipe.io.metadata` package provides functions for generating standard CTA
metadata headers and attaching them to output files.


Reference/API
=============

.. automodapi:: ctapipe.io
    :no-inheritance-diagram:

.. automodapi:: ctapipe.io.tableio
    :no-inheritance-diagram:

.. automodapi:: ctapipe.io.tableloader
    :no-inheritance-diagram:

.. automodapi:: ctapipe.io.hdf5tableio
    :no-inheritance-diagram:

.. automodapi:: ctapipe.io.metadata
    :no-inheritance-diagram:

.. automodapi:: ctapipe.io.eventsource
    :no-inheritance-diagram:

.. automodapi:: ctapipe.io.simteleventsource
    :no-inheritance-diagram:

.. automodapi:: ctapipe.io.hdf5eventsource
    :no-inheritance-diagram:

.. automodapi:: ctapipe.io.eventseeker
    :no-inheritance-diagram:
