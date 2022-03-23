.. _camera_readout:

.. currentmodule:: ctapipe.instrument

Camera Readout
==============

The `CameraReadout` stores information regarding the waveform readout from the
Cherenkov camera, such as sampling rate and information on the reference pulse
shape.

There are several ways to obtain a `CameraReadout`:

* Through the `SubarrayDescription` of an `~ctapipe.io.EventSource`

* use the `CameraReadout` constructor, where one has to specify all
  necessary information

* use ``CameraReadout.from_name(telescope, revision)`` (ex: ``readout =
  CameraReadout.from_name('HESS', 1)``).  This tries to read the corresponding data
  from through the service data utilities.

* load it from a pre-written file (which can be in any format
  supported by ``astropy.table.Table``, as long as that format allows for
  header-keywords as well as table entries.


`CameraReadout` is used by the `~ctapipe.image.extractor.ImageExtractor` in the `ctapipe.image` module
to ensure the images are scaled to the correct units.

Input/Output
------------


You can write out a `CameraReadout` by using the `CameraReadout.to_table()`
 method to turn it into an ``astropy.table.Table``, and then call its ``write()``
  function.  Reading it back in can be done with ``CameraReadout.from_table()``

.. code-block:: python

   readout = CameraReadout(...)  # constructed elsewhere

   readout.to_table().write('mycam.fits.gz') # FITS output
   readout.to_table().write('mycam.h5', path='/cameras/mycam') # hdf5 output
   readout.to_table().write('mycam.ecsv', format='ascii.ecsv') # text table

   # later read back in:

   readout = CameraReadout.from_table('mycam.ecsv', format='ascii.ecsv')
   readout = CameraReadout.from_table('mycam.fits.gz')
   readout = CameraReadout.from_table('mycam.h5', path='/cameras/mycam')


Reference/API
=============

.. automodapi:: ctapipe.instrument.camera.readout
    :no-inheritance-diagram:
