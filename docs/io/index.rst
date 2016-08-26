.. _io:

==============
 Input/Output
==============

.. currentmodule:: ctapipe.reco

Introduction
============

`ctapipe.io` contains functions and classes related to reading data,
like `~ctapipe.io.hessio_event_source`.  It currently contains some
data structures related to the observatory layout, like
:class:`~ctapipe.io.camera.CameraGeometry`, but those will likely move
soon to an instrument package.


Acces to Raw Data
=================

The `ctapipe.io.hessio` package provides a convenient wrapper to
reading *simtelarray* data files, like those used in CTA monte-carlo
productions. It requires the `pyhessio` package to be installed (see
:ref:`getting_started` for instructions installing `pyhessio`).

.. figure:: shower.png
	    
   an image read from a *simtelarray* data file.

..

Reference/API
=============

.. automodapi:: ctapipe.io.camera
    :no-inheritance-diagram:

------------------------------
       
.. automodapi:: ctapipe.io.hessio

------------------------------
		
.. automodapi:: ctapipe.io.containers
    :no-inheritance-diagram:



