Code Guidelines
===============

Coding should follow the CTA coding guidelines when possible (those
are defined in a separate document form the SYS group)

Here, we list useful guidelines for the logical structure of code (see
also the style-guide for code style)

Useful references for good coding practices:
--------------------------------------------

* http://swcarpentry.github.io/slideshows/best-practices/index.html
* http://arxiv.org/abs/1210.0530

Data Structures
---------------

Python is very flexible with data structures: data can be in classes,
dictionaries, lists, tuples, and numpy `NDArrays`.  Furthermore, the
structure of a class or dict is flexible: members can be added or
removed at runtime.  Therefore, we should be careful to follow some
basic guidelines:

* basic array-like data should be in an `numpy.NDArray`, with a suitable
  `dtype` (fixed data type)

* for complex sets of arrays, each with a column name and unit, can be
  managed via an `astropy.table.Table` object (which also provides
  methods to read and write such a table to/from nearly any file
  format, including FITS and HDF5). Other packages like `pandas` or
  `dask` may be explored.

* `dict` s or classes can be used for more flexible containers

* Classes don't need to sub-class `object`, because we only support
  Python 3 and new-style classes are the default, i.e. subclassing
  `object` is superfluous.
