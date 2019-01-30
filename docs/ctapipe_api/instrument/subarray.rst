.. _subarray_description:

.. currentmodule:: ctapipe.instrument

Subarray Description
====================

The `SubarrayDescription` class holds info about the list of telescopes in an
 array or subarray and  the position on the ground of each. Internally these
 are stored as dicts by `tel_id`.

You can get a quick look at an array with the `SubarrayDescription.info()`
and `SubarrayDescription.peek()` methods

You can also convert them to `astropy.table.Table` objects using
`SubarrayDescription.to_table()`, which can then be saved to disk or
manipulated.

