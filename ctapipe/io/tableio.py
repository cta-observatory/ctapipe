import re
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import numpy as np
from astropy.time import Time
from astropy.units import Quantity

from ..core import Component


__all__ = ["TableReader", "TableWriter"]


class TableWriter(Component, metaclass=ABCMeta):
    """
    Base class for writing  Container classes as rows of an output table,
    where each `Field` becomes a column. Subclasses of this implement
    specific output types.

    See Also
    --------
    ctapipe.io.HDF5TableWriter: Implementation of this base class for writing HDF5 files
    """

    def __init__(self, parent=None, add_prefix=False, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self._transforms = defaultdict(dict)
        self._exclusions = defaultdict(list)
        self.add_prefix = add_prefix

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def exclude(self, table_name, pattern):
        """
        Exclude any columns (Fields)  matching the pattern from being written

        Parameters
        ----------
        table_name: str
            name of table on which to apply the exclusion
        pattern: str
            regular expression string to match column name
        """
        table_name = table_name.lstrip("/")
        self._exclusions[table_name].append(re.compile(pattern))

    def _is_column_excluded(self, table_name, col_name):
        for pattern in self._exclusions[table_name]:
            if pattern.fullmatch(col_name):
                return True
        return False

    def add_column_transform(self, table_name, col_name, transform):
        """
        Add a transformation function for a column. This function will be
        called on the value in the container before it is written to the
        output file.

        Parameters
        ----------
        table_name: str
            identifier of table being written
        col_name: str
            name of column in the table (or item in the Container)
        transform: callable
            function that take a value and returns a new one
        """
        # allow leading slash
        self._transforms[table_name.lstrip("/")][col_name] = transform
        self.log.debug(
            "Added transform: {}/{} -> {}".format(table_name, col_name, transform)
        )

    @abstractmethod
    def write(self, table_name, containers, **kwargs):
        """
        Write the contents of the given container or containers to a table.
        The first call to write  will create a schema and initialize the table
        within the file.
        The shape of data within the container must not change between calls,
        since variable-length arrays are not supported.

        Parameters
        ----------
        table_name: str
            name of table to write to
        containers: ctapipe.core.Container or iterable thereof
            container instance(s) to write
        **kwargs:
            may be passed to a lower level implementation to set options
        """
        pass

    @abstractmethod
    def open(self, filename, **kwargs):
        """
        open an output file

        Parameters
        ----------
        filename: str
            output file name
        kwargs:
            any extra args to pass to the subclass open method
        """
        pass

    @abstractmethod
    def close(self):
        pass

    def _apply_col_transform(self, table_name, col_name, value):
        """
        apply value transform function if it exists for this column
        """
        if col_name in self._transforms[table_name]:
            tr = self._transforms[table_name][col_name]
            value = tr(value)
        return value


class TableReader(Component, metaclass=ABCMeta):
    """
    Base class for row-wise table readers. Generally methods that read a
    full table at once are preferred to this method, since they are faster,
    but this can be used to re-play a table row by row into a
    `ctapipe.core.Container` class (the opposite of TableWriter)
    """

    def __init__(self):
        super().__init__()
        self._cols_to_read = defaultdict(dict)
        self._transforms = defaultdict(dict)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def add_column_transform(self, table_name, col_name, transform):
        """
        Add a transformation function for a column. This function will be
        called on the value in the container before it is written to the
        output file.

        Parameters
        ----------
        table_name: str
            identifier of table being written
        col_name: str
            name of column in the table (or item in the Container)
        transform: callable
            function that take a value and returns a new one
        """
        self._transforms[table_name][col_name] = transform
        self.log.debug(
            "Added transform: {}/{} -> {}".format(table_name, col_name, transform)
        )

    def _apply_col_transform(self, table_name, col_name, value):
        """
        apply value transform function if it exists for this column
        """
        if col_name in self._transforms[table_name]:
            tr = self._transforms[table_name][col_name]
            value = tr.inverse(value)
        return value

    @abstractmethod
    def read(self, table_name, containers, prefixes, **kwargs):
        """
        Returns a generator that reads the next row from the table into the
        given container.  The generator returns the same container. Note that
        no containers are copied, the data are overwritten inside.

        Parameters
        ----------
        table_name: str
            name of table to read from
        containers: ctapipe.core.Container or iterable thereof
            Container instance(s) to fill
        prefixes: bool, str or iterable of str
            prefixes used during writing of the table
        """
        pass

    @abstractmethod
    def open(self, filename, **kwargs):
        pass

    @abstractmethod
    def close(self):
        pass


class ColumnTransform(metaclass=ABCMeta):
    """
    A Transformation to be applied before serialization / after deserialization.

    The ``TableWriter`` will call the transform on the data to be stored and
    ``TableReader`` will call `.inverse`` the reverse the transformation 
    when a transformation is detected in the file through metadata.

    Transformations implement ``get_meta`` to provide the necessary metadata
    for inverting the transformation on reading.
    """

    @abstractmethod
    def __call__(self, value):
        pass

    def inverse(self, value):
        """No inverse transform by default"""
        return value

    def get_meta(self, colname):
        """Empty meta by default"""
        return {}


class TimeColumnTransform(ColumnTransform):
    """A Column transformation that converts astropy time objects to MJD TAI"""

    def __init__(self, scale, format):
        self.scale = scale
        self.format = format

    def __call__(self, value: Time):
        """
        Convert an astropy time object to an mjd value in tai scale
        """
        return getattr(getattr(value, self.scale), self.format)

    def inverse(self, value):
        return Time(value, scale=self.scale, format=self.format, copy=False)

    def get_meta(self, colname):
        return {
            f"{colname}_TRANSFORM": "time",
            f"{colname}_TIME_FORMAT": self.format,
            f"{colname}_TIME_SCALE": self.scale,
        }


class QuantityColumnTransform(ColumnTransform):
    """ A Column Transform that transforms quantities to their values in the given unit"""

    def __init__(self, unit):
        self.unit = unit

    def __call__(self, value):
        return value.to_value(self.unit)

    def inverse(self, value):
        return Quantity(value, self.unit, copy=False)

    def get_meta(self, colname):
        return {
            f"{colname}_TRANSFORM": "quantity",
            f"{colname}_UNIT": self.unit.to_string("vounit"),
        }


class FixedPointColumnTransform(ColumnTransform):
    """
    Apply a scale, offset and dtype conversion.

    Can be used to store values as fixed point by using an integer dtype
    and a scale that is a power of 10.
    """

    def __init__(self, scale, offset, source_dtype, target_dtype):
        self.scale = scale
        self.offset = offset
        self.source_dtype = np.dtype(source_dtype)
        self.target_dtype = np.dtype(target_dtype)

    def __call__(self, value):
        return (value * self.scale).astype(self.target_dtype) + self.offset

    def inverse(self, value):
        return (value - self.offset).astype(self.source_dtype) / self.scale

    def get_meta(self, colname: str):
        return {
            f"{colname}_TRANSFORM": "fixed_point",
            f"{colname}_TRANSFORM_SCALE": self.scale,
            f"{colname}_TRANSFORM_DTYPE": str(self.source_dtype),
            f"{colname}_TRANSFORM_OFFSET": self.offset,
        }


class EnumColumnTransform(ColumnTransform):
    """Store the value of an enum"""

    def __init__(self, enum):
        self.enum = enum

    @staticmethod
    def __call__(value):
        return value.value

    def inverse(self, value):
        return self.enum(value)

    def get_meta(self, colname):
        return {f"{colname}_TRANSFORM": "enum", f"{colname}_ENUM": self.enum}
