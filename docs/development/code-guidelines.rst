Code Guidelines
===============

Here, we list useful guidelines for the logical structure of code (see
also the style-guide for code style)

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
  format, including FITS and HDF5)

* `dict` s or classes can be used for more flexible containers

* for cases where we need to have a data structure with a well-defined
  fixed set of attributes, a `collections.namedtuple` should be used
  to define the structure.  Namedtuples are like a _struct_ in
  C/C++: they must have all attributes filled in, attributes cannot be
  added or removed, and the names are fixed.  Unlike C/C++ structs,
  namedtuples are _immutable_, meaning their data cannot be changed
  after they are constructed. 

* Classes don't need to sub-class `object`, because we only support Python 3
  and new-style classes are the default, i.e. subclassing `object` is superfluous.
