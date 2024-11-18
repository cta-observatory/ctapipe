.. _camera_geometries:

.. currentmodule:: ctapipe.instrument

*****************
Camera Geometries
*****************

The `~ctapipe.instrument.CameraGeometry` provides an easy way to work with images or data
cubes related to Cherenkov Cameras.  In *ctapipe*, a camera image is
simply a flat 1D array (or 2D if time information is included), where
there is one value per pixel. Of course, to work with such an array,
one needs spatial information about how the pixels are laid out.
Since CTAO has at least 6 different camera types, and may have multiple
versions of each as revisions are made, it is necessary to have a
common way to describe all cameras.

So far there are several ways to construct a `~ctapipe.instrument.CameraGeometry`:

* `~ctapipe.io.EventSource` instances have a :attr:`~ctapipe.io.EventSource.subarray` attribute,
  e.g. to obtain the geometry for the telescope with id 1, use:
  ``source.subarray.tel[1].camera.geometry``. The
  `~ctapipe.io.TableLoader` instance also has the ``.subarray`` attribute.

* use the `~ctapipe.instrument.CameraGeometry` constructor, where one has to specify all
  necessary information (pixel positions, types, areas, etc)

* load it from a pre-written file (which can be in any format
  supported by `astropy.table`, as long as that format allows for
  header-keywords as well as table entries.


Once loaded, the `~ctapipe.instrument.CameraGeometry` object gives you access the pixel
positions, areas, neighbors, and shapes.

`~ctapipe.instrument.CameraGeometry` is used by most image processing algorithms in the
`ctapipe.image` module, as well as displays in the
`ctapipe.visualization` module.


Input/Output
============

You can write out a `~ctapipe.instrument.CameraGeometry` by using the `CameraGeometry.to_table()`
method to turn it into an `astropy.table.Table`, and then call its `~astropy.table.Table.write`
function.  Reading it back in can be done with `~ctapipe.instrument.CameraGeometry.from_table()`

.. code-block:: python

   geom = ~ctapipe.instrument.CameraGeometry(...)  # constructed elsewhere

   geom.to_table().write('mycam.fits.gz') # FITS output
   geom.to_table().write('mycam.h5', path='/cameras/mycam') # hdf5 output
   geom.to_table().write('mycam.ecsv', format='ascii.ecsv') # text table

   # later read back in:

   geom = ~ctapipe.instrument.CameraGeometry.from_table('mycam.ecsv', format='ascii.ecsv')
   geom = ~ctapipe.instrument.CameraGeometry.from_table('mycam.fits.gz')
   geom = ~ctapipe.instrument.CameraGeometry.from_table('mycam.h5', path='/cameras/mycam')


A Note On Pixel Neighbors
=========================

The `~ctapipe.instrument.CameraGeometry` object provides two pixel-neighbor
representations: a *neighbor adjacency list* (in the :attr:`~CameraGeometry.neighbors`
attribute) and a *pixel adjacency matrix* (in the :attr:`~CameraGeometry.neighbor_matrix`
attribute).  The former is a list of lists, where element *i* is a
list of neighbors *j* of the *i*th pixel.  The latter is a 2D matrix
where row *i* is a boolean mask of pixels that are neighbors. It is
not necessary to load or specify either of these neighbor
representations when constructing a `~ctapipe.instrument.CameraGeometry`, since they
will be computed on-the-fly if left blank, using a KD-tree
nearest-neighbor algorithm.

It is recommended that all algorithms that need to be computationally
fast use the :attr:`~CameraGeometry.neighbor_matrix` attribute, particularly in conjunction
with `numpy` operations, since it is quite speed-efficient.


Examples
========

.. plot:: api-reference/instrument/camerageometry_example.py
    :include-source:


See also `ctapipe.image.tailcuts_clean()` and `ctapipe.image.dilate()`
for usage examples.


Reference/API
=============

.. automodapi:: ctapipe.instrument.camera.geometry
    :no-inheritance-diagram:
