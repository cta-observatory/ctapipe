.. _io_unofficial:

======================================================
Unofficial Event Data Format Readers (`io.unofficial`)
======================================================

.. currentmodule:: ctapipe.io.unofficial


Introduction
============

As the official data format for CTA is still undefined, individual camera
protoptype teams have created their own data formats in which
their camera events are stored.

For data analysis, MC verification, and increasing familiarity with using
`ctapipe`, it is desirable to allow reading of all these data formats into
`ctapipe`. However it is important to keep in mind the following points in the
creation of a new unofficial reader:

* These readers are unofficial and temporary - every camera team will be
  adopting the official CTA data format once it is defined.
* Maintaining all these readers is expensive and inefficient for the long-term.
  Once cameras have transitioned to the new data format these unofficial
  readers will be removed from `ctapipe`. Experts who wish to read the old data
  formats are advised to either convert their old data into the official
  format, or locally restore the unofficial `ctapipe` reader for that format.

In an attempt to simplify and standardise the process of creating a new
reader, this documentation has been created, containing the guidelines to be
followed. These guidelines can be summarised as:

* Each reader should have a corresponding
  `ctapipe.io.eventfilereader.EventFileReader` in
  `ctapipe.io.unofficial.eventfilereader` to ensure the common high-level
  interface for reading data is maintained.
* Care should be taken in the imports of external modules. There should be no
  dependencies added to ctapipe on the external libraries required for these
  unofficial readers.


How to use my reader in ctapipe?
================================

Presumably you already have some software that exists to read in your data
format. What form that reader exists in dictates how easy it is to incorporate
into ctapipe. Some of the possible forms and appropriate inplementations are:

* **Your reader can be entirely programmed in Python**, in which case your
  reader does not depend on any external packages, and you can include that
  code in your reader's :ref:`Low-Level Reader-Specific Modules`.
* **Your reader exists as an external Python module**, in which case it could
  be made available on conda for easy (optional) installion.
* **You have a C/C++ inplementation of your reader in a simple script**,
  therefore you could provide an interface to that functionality by using
  "ctypes". The C/C++ scripts can exist inside your reader module, and are
  compiled as part of the ctapipe installation. See
  `ctapipe.utils.neighbour_sum` as an example of how to interface Python with
  C code, and remember to add your extension to ext_modules in setup.py.
* **Your reader exists as an external C/C++ library**, therefore software such
  as SWIG can be used to provide a Python wrapper to the library,
  automatically creating interfaces to the functions in the library.


EventFileReaders (`ctapipe.io.unofficial.eventfilereader`)
==========================================================

This class provides the high-level interface for reading data from each data
format. Read about the purpose of EventFileReaders at
:ref:`EventFileReaderClasses`.

Location for every unofficial `ctapipe.io.eventfilereader.EventFileReader`.

Each reader is required to import their
:ref:`Low-Level Reader-Specific Modules` inside their `__init__` method,
therefore bypassing the need for the external reader software to be installed
unless someone has chosen to use this specific
`ctapipe.io.eventfilereader.EventFileReader`, in which case a
`ModuleNotFoundError` will be raised if the external software has not been
found.

The reader must also implement a fast method for
`ctapipe.io.eventfilereader.EventFileReader.check_file_compatibility`. This
method attempts to find a reader which *may* be compatible with the input
file, and can be as simple as looking for a unique file extension in the file
path or attempting to find a header keyword in the file. It is strongly
advised that this method does not depend on the external reader software, as
this would introduce a dependency when choosing the correct reader in the
`ctapipe.io.eventfilereader.EventFileReaderFactory`.

The currently existing EventFileReaders,
`ctapipe.io.eventfilereader.HessioFileReader` and
`ctapipe.io.unofficial.eventfilereader.TargetioFileReader` can be used as
templates to create a new EventFileReader for a different data format.


Low-Level Reader-Specific Modules
===========================

A seperate module should be created for the low-level method used for filling
the event containers for a specific data format. The naming convention for
this module is "\*io.py", where "\*" is the short-name for the data format.
For example: `ctapipe.io.hessio` and `ctapipe.io.unofficial.targetio.targetio`.

The low-level module can be a directory of multiple modules, allowing the
various components involved in the event building (such as containers) to be
grouped together for a particular data format.

Each data format can be wildly different, therefore these low-level
modules can appear very different in how they function. This is fine as long
as these rules are followed:

* The `ctapipe.io.eventfilereader.EventFileReader` calls the low-level methods
  such that the high-level interface remains the same as other
  `ctapipe.io.eventfilereader.EventFileReader`.
* The low-level methods fill the containers with the relevant data, e.g. the
  waveforms are stored in "data.r0.tel[].adc_samples" (assuming they are
  waveforms of data level R0, see the next section for an explanation of data
  levels and see `ctapipe.io.hessio` and
  `ctapipe.io.unofficial.targetio.targetio` for examples)
* The unique 'origin' string is added to the meta dictionary of the container
  so the data source of the events can be obtained by other algorithms, e.g.

  >>> data.meta['origin'] = "targetio"


Data Levels (R0/R1/DL0/DL1)
===========================

It is important to read your events into the correct corresponding event
container:

* If the file contains raw events, that still require low-level calibration
  to be useful for offline analysis, then they should be read into the r0
  container.
* If the file contains events which have their low-level calibration fully
  applied to them, then they should be read into the r1 container.
* If the file contains events which have their low-level calibration fully
  applied to them AND they are in a proposed format of DL0 (perhaps including
  a data reduction approach), then they should be stored in the dl0 container.
* If the file contains events which are already reduced into an extracted
  charge per pixel, then they should be stored in the dl1 container.

See this link for more details about the `CTA High-Level Data Model
Definitions SYS-QA/160517
<https://jama.cta-observatory.org/perspective.req?projectId=6&docId=26528>`_
(CTA internal).


Containers and R1 Calibration
=============================

This is an optional section. It is included here because the R1 calibration
is the main other inclusion camera teams may need to provide to ctapipe for
their prototype data.

If the events stored in the file are at the R0 data level, then you may need
to create an `ctapipe.calib.camera.r1.CameraR1Calibrator` to perform the
low-level camera-specific R1 calibration required for your events.

At the R0 level the data contents are very camera specific - there may be
additional information at this level which is used in the R1 calibration, but
then is unused afterwards. Therefore it may be required to create a unique R0
container for that data format. This container must contain the same
information as `ctapipe.io.containers.R0CameraContainer`, but can contain
additional information which may be used by the
`ctapipe.calib.camera.r1.CameraR1Calibrator`. An example of this for
the TargetioEventReader is the `first_cell_ids` field in
`ctapipe.io.unofficial.targetio.containers.TargetioR0CameraContainer`.

Additionally, to automate the selection of the correct
`ctapipe.calib.camera.r1.CameraR1Calibrator` from
`ctapipe.calib.camera.r1.CameraR1CalibratorFactory`, the
`ctapipe.io.eventfilereader.EventFileReader.r1_calibrator` property can be
used to specify the correct calibrator to use.
This is then obtained from the reader, and passed to the
`ctapipe.calib.camera.r1.CameraR1CalibratorFactory`. An example of this can be
found in ctapipe/examples/calibration_pipeline.py. If a particular data format
is to be used by multiple cameras, each requiring different R1 calibration,
then this method should correctly return the right one. By default, a new
reader will return 'NullR1Calibrator'.


Instrument Configuration
========================

As the `ctapipe.instrument` module and "instrument database" are not fully
developed, the camera configuration information (such as pixel positions) are
still usually obtained from inside the hessio file.

Data formats other than hessio are unlikely to contain the camera
configuration, therefore until the `ctapipe.instrument` module and the
"instrument database" are completed, the unofficial readers must contain other
methods to load the instrument information into the container. This can
involve loading data from additional files, as long as these data files are
added to the `data_files` dictionary in setup.py.

For an example of one approach to achieve this, see
`ctapipe.io.unofficial.targetio.camera`.


Tests and Documentation
=======================

Every unofficial reader must be thoroughly tested and documented in their
docstrings. This is especially crucial for these modules due to their
uniqueness and differences between one reader and another.

As we want to avoid adding new depencies to ctapipe, one should make tests of
their EventFileReader skippable if the external dependencies do not exist on
the machine. This can be done by either:

>>> import pytest
>>> pytest.importorskip("external_module")

To skip all the tests in the file, or:

>>> import pytest
>>> def test_something():
>>>    pytest.importorskip("external_module")

To skip a single test function.

The method to install the external software should be added to the .travis.yml
file, so that the Travis CI can perform all tests, including the unofficial
reader tests. Additionally, a small test file that is in an unofficial data
format should be committed to ctapipe-extra.

Installation instructions for external software should be included in the
docstrings of the EventFileReader.


TargetioFileReader
==================

`ctapipe.io.unofficial.eventfilereader.TargetioFileReader` is the file reader
for the data format used by cameras containing TARGET modules, such as CHEC
for the GCT SST. It provides a template for how to create an EventFileReader
for other unofficial data formats.

To read and calibrate these files, we use three external libraries:

* TargetDriver
* TargetIO
* TargetCalib

These are C++ libraries that have been wrapped by SWIG to produce the python
modules `target_driver`, `target_io`, and `target_calib`.

This reader is set-up to handle both R0 and R1 data level files (either can be
stored in this data format). The data level that is stored in a targetio file
is determined by looking at a flag in the file header.

The files that are compatible with this reader are stored with the extension
".tio", which is checked for by
`ctapipe.io.eventfilereader.EventFileReader.check_file_compatibility`.

The installation instructions for the external libraries required for
TargetioFileReader can be found at the following link:
https://forge.in2p3.fr/projects/gct/wiki/Installing_CHEC_Software


Reference/API
=============

.. automodapi:: ctapipe.io.unofficial.eventfilereader

------------------------------

.. automodapi:: ctapipe.io.unofficial.targetio

------------------------------

.. automodapi:: ctapipe.io.unofficial.targetio.targetio

------------------------------

.. automodapi:: ctapipe.io.unofficial.targetio.camera

------------------------------

.. automodapi:: ctapipe.io.unofficial.targetio.containers