.. _camera_geometries:

.. currentmodule:: ctapipe.instrument

Camera Geometries
=================

The `CameraGeometry` provides an easy way to work with images or data
cubes related to Cherenkov Cameras.  In *ctapipe*, a camera image is
simply a flat 1D array (or 2D if time information is included), where
there is one value per pixel. Of course, to work with such an array,
one needs spatial information about how the pixels are laid out.
Since CTA has at least 6 different camera types, and may have multiple
versions of each as revisions are made, it is necessary to have a
common way to describe all cameras.

So far there are several ways to construct a `CameraGeometry`:

* use the `CameraGeometry` constructor, where one has to specify all
  necessary information (pixel positions, types, areas, etc)

* use `CameraGeometry.from_name(telescope, revision)` (ex: `geom =
  CameraGeometry.from_name('HESS',1)`).  This reads the telescope def
  from the `$CTAPIPE_EXTRA` directory, and so far we only have HESS
  telescopes there (more to come)

* load a Monte-Carlo file, get the list of pixel X and Y positions and
  the telescope focal length and use
  `CameraGeometry.guess(x,y,flen)` - this will work for all telescopes
  in CTA so far

* load it from a pre-written file (which can be in any format
  supported by `astropy.table`, as long as that format allows for
  header-keywords as well as table entries.


Once loaded, the `CameraGeometry` object gives you access the pixel
positions, areas, neighbors, and shapes.  Since the geometries are
cached in the class, subsequent calls to `CameraGeometry.guess()` will
return the same object if the inputs are the same, and are thus
speed-efficient.

`CameraGeometry` is used by most image processing algorithms in the
`ctapipe.image` module, as well as displays in the
`ctapipe.visualization` module.

Input/Output
------------

You can write out a `CameraGeometry` by using the `CameraGeometry.to_table()`
 method to turn it into an `astropy.table.Table`, and then call its `write()`
  function.  Reading it back in can be done with `CameraGeometry.from_table()`

.. code-block:: python

   geom = CameraGeometry(...)  # constructed elsewhere

   geom.to_table().write('mycam.fits.gz') # FITS output
   geom.to_table().write('mycam.h5', path='/cameras/mycam') # hdf5 output
   geom.to_table().write('mycam.ecsv', format='ascii.ecsv') # text table

   # later read back in:

   geom = CameraGeometry.from_table('mycam.ecsv', format='ascii.ecsv')
   geom = CameraGeometry.from_table('mycam.fits.gz')
   geom = CameraGeometry.from_table('mycam.h5', path='/cameras/mycam')


A note on Pixel Neighbors
-------------------------

The `CameraGeometry` object provides two pixel-neighbor
representations: a *neighbor adjacency list* (in the `neighbors`
attribute) and a *pixel adjacency matrix* (in the `neighbor_matrix`
attribute).  The former is a list of lists, where element *i* is a
list of neighbors *j* of the *i*th pixel.  The latter is a 2D matrix
where row *i* is a boolean mask of pixels that are neighbors. It is
not necessary to load or specify either of these neighbor
representations when constructing a `CameraGeometry`, since they
will be computed on-the-fly if left blank, using a KD-tree
nearest-neighbor algorithm.

It is recommended that all algorithms that need to be computationally
fast use the `neighbor_matrix` attribute, particularly in conjunction
with `numpy` operations, since it is quite speed-efficient.

Examples
--------

.. plot:: instrument/camerageometry_example.py
    :include-source:


see also `ctapipe.image.tailcuts_clean()` and `ctapipe.image.dilate()`
for usage examples

