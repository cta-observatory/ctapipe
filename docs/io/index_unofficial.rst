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
`ctapipe`. However, care must be taken with this process:

* Each reader should have a corresponding
  `ctapipe.io.eventfilereader.EventFileReader` in
  `ctapipe.io.unofficial.eventfilereader` to ensure the common high-level
  interface for reading data is maintained.
* Care should be taken in the imports of external modules. There should be no
  dependencies added to ctapipe on the external libraries required for these
  unofficial readers. Each `ctapipe.io.eventfilereader.EventFileReader`
  should be conditionally instanced provided the required external libraries
  are already installed on the machine.
* These readers are unofficial and temporary - every camera team will be
  adopting the official CTA data format once it is defined.
* Maintaining all these readers is expensive and inefficient for the long-term.
  Once cameras have transitioned to the new data format these unofficial
  readers will be removed from `ctapipe`. Experts who wish to read the old data
  formats are advised to either convert their old data into the official
  format, or locally restore the unofficial `ctapipe` reader for that format.


EventFileReaders (`ctapipe.io.unofficial.eventfilereader`)
==========================================================

Location for every unofficial `ctapipe.io.eventfilereader.EventFileReader`.

Each reader should utilise `ctapipe.utils.check_modules_installed` to
conditionally instance the `ctapipe.io.eventfilereader.EventFileReader` if the
required external libraries exist on the machine. The imports of any low-level
methods should also be conditionally imported here too. This ensures that no
errors are raised if a machine does not have these optional dependencies
installed.

The reader should also implement a fast method for
`ctapipe.io.eventfilereader.EventFileReader.check_file_compatibility`, which
can be as simple as looking at for a unique file extension in the file name,
or calling a method from the external reader library to check that the file is
compatible.

Additionally, each EventFileReader should contain a property to return the
name of the `ctapipe.calib.camera.r1.CameraR1Calibrator` that should be used
for calibrating the events. The CameraR1Calibrator that should be chosen
depends on the source of the events. If a data format is to be used by
different cameras that require different R1 calibration, then this method
should correctly figure out the right one. If the data source is from data
level R1 or above, return None or 'NullR1Calibrator'.

These new EventFileReaders should follow the
template of `ctapipe.io.eventfilereader.HessioFileReader` and
`ctapipe.io.unofficial.eventfilereader.TargetioFileReader`.


Low-Level Reader-Specific Modules
===========================

A seperate module should be created for the low-level method used for filling
the event containers for a specific data format. The naming convention for
this module is "*io.py", where "*" is the short-name for the data format.

Each reader may have multiple files required in the full building of the event
containers. Therefore the low-level module can be a directory of multiply
modules.

Each unofficial data format can be wildly different, therefore these low-level
modules can appear very different in how they function. This is fine as long
as some guidelines are followed:

* The `ctapipe.io.eventfilereader.EventFileReader` calls the low-level methods
  such that the high-level interface remains the same as other
  `ctapipe.io.eventfilereader.EventFileReader`.
* The low-level methods fill the containers with the correct data, in a
  similar way to other low-level reader methods (see `ctapipe.io.hessio` and
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

If the events stored in the file are at the R0 data level, then you must also
create an `ctapipe.calib.camera.r1.CameraR1Calibrator` to perform the
low-level calibration required for your events.

At the R0 level the data contents is very camera specific - there may be
additional information at this level which is used in the R1 calibration, but
then is unused afterwards. Therefore it may be required to create a unique R0
container for that data format, containing additional imformation that is used
by the `ctapipe.calib.camera.r1.CameraR1Calibrator`. An example of this for
the TargetioEventReader is the `first_cell_ids` in
`ctapipe.io.unofficial.targetio.containers`.


Instrument Configuration
========================

As the `ctapipe.instrument` module and "instrument database" are not fully
developed, the camera configuration information (such as pixel positions) are
still usually obtained from inside the hessio file.

Data formats other than hessio are unlikely to contain the camera
configuration, therefore until the `ctapipe.instrument` module and the
"instrument database" are completed, the unofficial readers must contain other
methods to load the instrument information into the container.

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
>>> @pytest.mark.skipif(not check_modules_installed(["external_modules"]),
>>>                     reason="Requires external unofficial reader modules")
>>> def test_something():

To skip a single test function.

The method to install the external software should be added to the .travis.yml
file, so that the Travis CI can perform all tests, including the unofficial
reader tests. Additionally, a small test file that uses in unofficial data
format should be committed to ctapipe-extra.

Installation instructions for external software should be included in the
docstrings of the EventFileReader.


TargetioFileReader
==================

`ctapipe.io.unofficial.eventfilereader.TargetioFileReader` is the file reader
for the data format used by cameras containing TARGET modules, such as CHEC
for the GCT SST. It provides a template for how to create an EventFileReader
for another unofficial data formats.

To read and calibrate these files, we use three external libraries:

* TargetDriver
* TargetIO
* TargetCalib

These are C++ libraries that have been wrapped by SWIG to produce the python
modules `target_driver`, `target_io`, and `target_calib`.

This reader is set-up to handle both R0 and R1 data levels (both stored in the
same data format). The data levels are differed by looking at a flag in the
header.

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