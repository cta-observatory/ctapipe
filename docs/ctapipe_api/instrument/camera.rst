.. _camera_description:

.. currentmodule:: ctapipe.instrument

Camera Description
==================

The `~CameraDescription` contains classes holding information about the
Cherenkov camera, namely the `~CameraGeometry` and `~CameraReadout` classes.


.. toctree::
  :maxdepth: 1

  camera_geometry
  camera_readout


Reference/API
=============

.. What follows is a temporary workaround to circumvent
   various warnings of duplicate references caused by
   calling automodapi on the camera package.

ctapipe.instrument.camera Package
---------------------------------


Classes
^^^^^^^

.. autosummary::

    ~CameraDescription
    ~CameraGeometry
    ~PixelShape
    ~camera.geometry.UnknownPixelShapeWarning
    ~CameraReadout


.. automodapi:: ctapipe.instrument.camera.description
    :no-inheritance-diagram:
