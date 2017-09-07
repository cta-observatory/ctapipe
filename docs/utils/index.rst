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

The `get_dataset()` function provides a common way to load CTA "SVC"
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

```python

from ctapipe.utils import get_table_dataset

optics = get_table_dataset('optics')
print(optics)

```

```

<Table masked=True length=7>
tel_description tel_type tel_subtype  mirror_area  mirror_type num_mirror_tiles effective_focal_length
                                           m2                                             m           
     str14        str3      str10       float64        str2         int64              float64        
--------------- -------- ----------- ------------- ----------- ---------------- ----------------------
        MST-SCT      MST         SCT 73.3475723267          SC                2          5.58629989624
        SST-GCT      SST         GCT 12.5663709641          SC                2          2.28299999237
            LST      LST          --  386.73324585          DC              198                   28.0
      SST-ASTRI      SST       ASTRI 14.5625667572          SC                2          2.15000009537
            MST      MST          -- 103.830558777          DC               84                   16.0
         SST-1M      SST          1M 9.41809272766          SC               18          5.59999990463
 MST-Whipple10m      MST  Whipple10m          75.0          DC              248                    7.3

```


Reference/API
=============

.. automodapi:: ctapipe.utils
    :no-inheritance-diagram:

.. automodapi:: ctapipe.utils.linalg
    :no-inheritance-diagram:

