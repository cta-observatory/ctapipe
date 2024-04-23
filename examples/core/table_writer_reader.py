"""
Writing Containers to a tabular format
======================================

The ``TableWriter``/``TableReader`` sub-classes allow you to write a
``ctapipe.core.Container`` class and its meta-data to an output table.
They treat the ``Field``s in the ``Container`` as columns in the
output, and automatically generate a schema. Here we will go through an
example of writing out data and reading it back with *Pandas*,
*PyTables*, and a ``ctapipe.io.TableReader``:

In this example, we will use the ``HDF5TableWriter``, which writes to
HDF5 datasets using *PyTables*. Currently this is the only implemented
TableWriter.

"""

######################################################################
# Caveats to think about: \* vector columns in Containers *can* be
# written, but some lilbraries like Pandas can not read those (so you must
# use pytables or astropy to read outputs that have vector columns)
# \* units are stored in the table metadata, but some libraries like Pandas
# ignore them and all other metadata
#


######################################################################
# Create some example Containers
# ------------------------------
#

import os

import numpy as np
import pandas as pd
import tables
from astropy import units as u

from ctapipe.core import Container, Field
from ctapipe.io import HDF5TableReader, HDF5TableWriter, read_table


######################################################################
class VariousTypesContainer(Container):
    a_int = Field(int, "some int value")
    a_float = Field(float, "some float value with a unit", unit=u.m)
    a_bool = Field(bool, "some bool value")
    a_np_int = Field(np.int64, "a numpy int")
    a_np_float = Field(np.float64, "a numpy float")
    a_np_bool = Field(np.bool_, "np.bool")


######################################################################
# let’s also make a dummy stream (generator) that will create a series of
# these containers
#


def create_stream(n_event):
    data = VariousTypesContainer()
    for i in range(n_event):
        data.a_int = int(i)
        data.a_float = float(i) * u.cm  # note unit conversion will happen
        data.a_bool = (i % 2) == 0
        data.a_np_int = np.int64(i)
        data.a_np_float = np.float64(i)
        data.a_np_bool = np.bool_((i % 2) == 0)

        yield data


######################################################################
for data in create_stream(2):
    for key, val in data.items():
        print("{}: {}, type : {}".format(key, val, type(val)))


######################################################################
# Writing the Data (and good practices)
# -------------------------------------
#


######################################################################
# Always use context managers with IO classes, as they will make sure the
# underlying resources are properly closed in case of errors:
#

try:
    with HDF5TableWriter("container.h5", group_name="data") as h5_table:
        for data in create_stream(10):
            h5_table.write("table", data)
            0 / 0
except Exception as err:
    print("FAILED:", err)
print("Done")

h5_table.h5file.isopen

######################################################################
print(os.listdir())

######################################################################
# Appending new Containers
# ------------------------
#


######################################################################
# To append some new containers we need to set the writing in append mode
# by using: ‘mode=a’. But let’s now first look at what happens if we
# don’t.
#

for i in range(2):
    with HDF5TableWriter(
        "container.h5", mode="w", group_name="data_{}".format(i)
    ) as h5_table:
        for data in create_stream(10):
            h5_table.write("table", data)

        print(h5_table.h5file)

######################################################################
os.remove("container.h5")

######################################################################
# Ok so the writer destroyed the content of the file each time it opens
# the file. Now let’s try to append some data group to it! (using
# mode=‘a’)
#

for i in range(2):
    with HDF5TableWriter(
        "container.h5", mode="a", group_name="data_{}".format(i)
    ) as h5_table:
        for data in create_stream(10):
            h5_table.write("table", data)

        print(h5_table.h5file)


######################################################################
# So we can append some data groups. As long as the data group_name does
# not already exists. Let’s try to overwrite the data group : data_1
#

try:
    with HDF5TableWriter("container.h5", mode="a", group_name="data_1") as h5_table:
        for data in create_stream(10):
            h5_table.write("table", data)
except Exception as err:
    print("Failed as expected:", err)


######################################################################
# Good ! I cannot overwrite my data.
#

print(bool(h5_table.h5file.isopen))


######################################################################
# Reading the Data
# ----------------
#


######################################################################
# Reading the whole table at once:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# For this, you have several choices. Since we used the HDF5TableWriter in
# this example, we have at least these options available:
#
# -  Pandas
# -  PyTables
# -  Astropy Table
#
# For other TableWriter implementations, others may be possible (depending
# on format)
#


######################################################################
# Reading using ``ctapipe.io.read_table``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# This is the preferred method, it returns an astropy ``Table`` and
# supports keeping track of units, metadata and transformations.
#


table = read_table("container.h5", "/data_0/table")
table[:5]

######################################################################
table.meta


######################################################################
# Reading with Pandas:
# ^^^^^^^^^^^^^^^^^^^^
#
# Pandas is a convenient way to read the output. **HOWEVER BE WARNED**
# that so far Pandas does not support reading the table *meta-data* or
# *units* for columns, so that information is lost!
#


data = pd.read_hdf("container.h5", key="/data_0/table")
data.head()


######################################################################
# Reading with PyTables
# ^^^^^^^^^^^^^^^^^^^^^
#


h5 = tables.open_file("container.h5")
table = h5.root["data_0"]["table"]
table


######################################################################
# note that here we can still access the metadata
#


table.attrs

######################################################################
# close the file
#
h5.close()


######################################################################
# Reading one-row-at-a-time:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# Rather than using the full-table methods, if you want to read it
# row-by-row (e.g. to maintain compatibility with an existing event loop),
# you can use a ``TableReader`` instance.
#
# The advantage here is that units and other metadata are retained and
# re-applied
#


def read(mode):
    print("reading mode {}".format(mode))

    with HDF5TableReader("container.h5", mode=mode) as h5_table:
        for group_name in ["data_0/", "data_1/"]:
            group_name = "/{}table".format(group_name)
            print(group_name)

            for data in h5_table.read(group_name, VariousTypesContainer):
                print(data.as_dict())


######################################################################
read("r")

######################################################################
read("r+")

######################################################################
read("a")

######################################################################
read("w")
