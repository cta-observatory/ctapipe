.. _utils:

===================
Utilities (`utils`)
===================

.. currentmodule:: ctapipe.utils

Introduction
============

`ctapipe.utils` contains a variety of low-level functionality used by
other modules that are not part of the `ctapipe.core` package.
Classes in this package may eventually move to `ctapipe.core` if they
have a stable enough API and are more widely used.

It currently provides:

* ND Histogramming (see `Histogram`)
* ND table interpolation (see `TableInterpolator`)
* access to service datasets
* linear algebra helpers
* dynamic class access
* json conversion


Access to Service Data files
============================

The `get_dataset_path()` function provides a common way to load CTA "SVC"
data (e.g. required lookups, example data, etc). It returns the full
directory path to the requested file. It currently works as follows:

1. it checks all directories in the CTA_SVC_PATH environment variable
   (which should be a colon-separated list of directories, like PATH)

2. if it doesn't find it there, it checks the ctapipe_resources module (which
   should be installed already in the package ctapipe-extra), which contains
   defaults.

Tabular data can be accessed automatically using
`get_table_dataset(basename)`, where the `basename` is the filename
without the extension.  Several tabular formats will be searched and
when the file is found it will be opened and read as an
`astropy.table.Table` object.  For example:

.. code-block:: python

    from ctapipe.utils import get_table_dataset

    optics = get_table_dataset('optics')
    print(optics)


Reference/API
=============

.. automodapi:: ctapipe.utils
    :no-inheritance-diagram:

.. automodapi:: ctapipe.utils.linalg
    :no-inheritance-diagram:

