"""
Base functionality for reading and writing tabular data
"""

import re
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import override

import numpy as np
from astropy.time import Time
from astropy.units import Quantity

from ctapipe.compat import COPY_IF_NEEDED

from ..core import Component
from ..instrument import SubarrayDescription
from ..time import ctao_high_res_to_time, time_to_ctao_high_res

__all__ = [
    "TableReader",
    "TableWriter",
    "ColumnTransform",
    "TimeColumnTransform",
    "QuantityColumnTransform",
    "FixedPointColumnTransform",
    "TelListToMaskTransform",
    "StringTransform",
    "EnumColumnTransform",
]


class TableWriter(Component, metaclass=ABCMeta):
    """
    Base class for writing tabular data stored in `~ctapipe.core.Container`

    A single write will add a new row to the output table,
    where each `~ctapipe.core.Field` becomes a column.
    Subclasses of this implement specific output types.

    Parameters
    ----------
    parent : ctapipe.core.Component or ctapipe.core.Tool or None, optional
        Parent from which to inherit configuration and logging.
        Mutually exclusive with ``config``.
    add_prefix : bool, optional
        Prefix column names with the container's prefix when writing.
    **kwargs
        Component configuration and trait values for the table writer.

    See Also
    --------
    ctapipe.io.HDF5TableWriter: Implementation of this base class for writing HDF5 files
    """

    def __init__(self, parent=None, add_prefix=False, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self._transform_regexps = defaultdict(dict)  # user requested, may be regexps
        self._transforms = defaultdict(dict)  # fully expanded explicit names
        self._exclusions = defaultdict(list)  # columns to exclude
        self.add_prefix = add_prefix

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def exclude(self, table_regexp, pattern):
        """Exclude matching fields when writing a table.

        The patterns are matched with re.fullmatch.

        Parameters
        ----------
        table_regexp : str
            regexp matching to match table name to apply exclusion
        pattern : str
            regular expression string to match column name in the table

        """
        table_regexp = table_regexp.lstrip("/")
        self._exclusions[re.compile(table_regexp)].append(re.compile(pattern))

    def _is_column_excluded(self, table_name, col_name):
        for table_regexp, col_regexp_list in self._exclusions.items():
            if table_regexp.fullmatch(table_name):
                for col_regexp in col_regexp_list:
                    if col_regexp.fullmatch(col_name):
                        return True
        return False

    def add_column_transform(self, table_name, col_name, transform):
        """Register a transform for one column.

        This function will be called on the value in the container before it is written
        to the output file.

        Parameters
        ----------
        table_name : str
            identifier of table being written
        col_name : str
            name of column in the table (or item in the Container).
        transform : ColumnTransform
            function that transforms input into output

        """
        # allow leading slash
        self._transforms[table_name.lstrip("/")][col_name] = transform
        self.log.debug(
            "Added transform: {}/{} -> {}".format(table_name, col_name, transform)
        )

    def add_column_transform_regexp(self, table_regexp, col_regexp, transform):
        """Register a transform for matching tables and columns.

        Table and column names are matched by regular expression. Matches are resolved
        into explicit column transforms when the table schema is built.

        Parameters
        ----------
        table_regexp : str
            Regular expression required to match the full table name.
        col_regexp : str
            Regular expression required to match the full column name in a matching
            table.
        transform : ColumnTransform
            function that tranformns input value into output value

        """
        # allow leading slash
        self._transform_regexps[table_regexp.lstrip("/")][col_regexp] = transform
        self.log.debug(
            "Requested transform for pattern: %s/%s -> %s",
            table_regexp,
            col_regexp,
            transform,
        )

    def _realize_regexp_transforms(self, table_name, containers):
        """Resolve transform patterns to specific fields for a table.

        For each matching field in the container, register an explicit transform with
        `add_column_transform`. This avoids checking the regular expressions for every
        table and column on each write.

        This should be called when building the table schema.

        Parameters
        ----------
        table_name : str
            table name
        containers : List[Container]
            List of containers to check
        """
        for table_regexp, column_regexp_dict in self._transform_regexps.items():
            if not re.fullmatch(table_regexp, table_name):
                continue

            self.log.debug("Table '%s' matched pattern '%s'", table_name, table_regexp)

            for column_regexp, transform in column_regexp_dict.items():
                for container in containers:
                    for col_name, _ in container.items(add_prefix=self.add_prefix):
                        if re.fullmatch(column_regexp, col_name):
                            self.log.debug(
                                "Column '%s' matched pattern '%s'",
                                col_name,
                                column_regexp,
                            )

                            self.add_column_transform(
                                table_name=table_name,
                                col_name=col_name,
                                transform=transform,
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
        table_name : str
            name of table to write to
        containers : ctapipe.core.Container or iterable thereof
            container instance(s) to write
        **kwargs
            may be passed to a lower level implementation to set options
        """
        pass

    @abstractmethod
    def open(self, filename, **kwargs):
        """
        Open an output file

        Parameters
        ----------
        filename : str
            output file name
        **kwargs
            any extra args to pass to the subclass open method
        """
        pass

    @abstractmethod
    def close(self):
        """Close open writer"""
        pass

    def _apply_col_transform(self, table_name, col_name, value):
        """
        Apply value transform function if it exists for this column
        """
        if col_name in self._transforms[table_name]:
            tr = self._transforms[table_name][col_name]
            value = tr(value)
        return value


class TableReader(Component, metaclass=ABCMeta):
    """
    Define the interface for reading table rows.

    Prefer reading the whole table at once when feasible for performance. This interface
    is useful for replaying rows as `ctapipe.core.Container` objects.
    """

    def __init__(self):
        super().__init__()
        self._transforms = defaultdict(dict)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def add_column_transform(self, table_name, col_name, transform):
        """
        Register a column transform for reading a table.

        The reader calls the transform's ``inverse`` method on stored values before
        placing them in a container.

        Parameters
        ----------
        table_name : str
            Name of the table being read.
        col_name : str
            Name of the column to transform.
        transform : ColumnTransform
            Transform whose inverse is applied to the stored value.
        """
        self._transforms[table_name][col_name] = transform
        self.log.debug(
            "Added transform: {}/{} -> {}".format(table_name, col_name, transform)
        )

    def _apply_col_transform(self, table_name, col_name, value):
        """
        Apply value transform function if it exists for this column
        """
        if col_name in self._transforms[table_name]:
            tr = self._transforms[table_name][col_name]
            value = tr.inverse(value)
        return value

    @abstractmethod
    def read(self, table_name, containers, prefixes, **kwargs):
        """
        Yield table rows as containers.

        The concrete reader determines how the containers are constructed.

        Parameters
        ----------
        table_name : str
            Name of the table to read.
        containers : type or sequence of types
            Container classes to populate.
        prefixes : bool, str, iterable of str or None
            How to match column prefixes used when writing the table.
        **kwargs
            Additional options supported by the concrete reader.
        """
        pass

    @abstractmethod
    def open(self, filename, **kwargs):
        """
        Open the input file for reading.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path of the input file.
        **kwargs
            Additional options for the file backend.
        """
        pass

    @abstractmethod
    def close(self):
        """Close the input file."""
        pass


class ColumnTransform(metaclass=ABCMeta):
    """
    A Transformation to be applied before serialization / after deserialization.

    The ``TableWriter`` will call the transform on the data to be stored and
    ``TableReader`` will call ``.inverse`` to reverse the transformation
    when a transformation is detected in the file through metadata.

    Transformations implement ``get_meta`` to provide the necessary metadata
    for inverting the transformation on reading.
    """

    @abstractmethod
    def __call__(self, value):
        """
        Convert a column value to its representation for storage.

        Parameters
        ----------
        value : object
            Python or Astropy value, such as an array, quantity, or timestamp.

        Returns
        -------
        object
            Encoded value suitable for the table backend.
        """
        pass

    @abstractmethod
    def inverse(self, value):
        """Invert the transformation applied in ``__call__``

        Parameters
        ----------
        value : object
            Stored column value to convert back to its in-memory representation.
        """
        return value

    @abstractmethod
    def get_meta(self, colname):
        """Metadata to be stored in the header information of the table

        Needs to fully describe the transformation so upon reading, the
        transformation can be inverted based on this metadata.

        Parameters
        ----------
        colname : str
            Column name used to form the keys of the transformation metadata.
        """
        return {}


class TimeColumnTransform(ColumnTransform):
    """A Column transformation that converts astropy time objects to a specific format.

    Parameters
    ----------
    scale : str
        Astropy time scale used with standard Time formats, such as ``"tai"``.
    format : str
        Astropy Time output format, or ``"ctao_high_res"`` for the CTAO time encoding.
    """

    def __init__(self, scale, format):
        self.scale = scale
        self.format = format

    def __call__(self, value: Time):
        """
        Convert Astropy times to the configured storage format.

        Parameters
        ----------
        value : astropy.time.Time
            Scalar or array of times to convert to the configured storage
            representation.
        """
        # ctao_high_res is not implemented as an astropy format but using
        # conversion functions defined in ctapipe
        if self.format == "ctao_high_res":
            return time_to_ctao_high_res(value)
        # for all other formats, use astropy mechanism
        return getattr(getattr(value, self.scale), self.format)

    @override
    def inverse(self, value):
        # ctao_high_res is not implemented as an astropy format but using
        # conversion functions defined in ctapipe
        if self.format == "ctao_high_res":
            return ctao_high_res_to_time(value[..., 0], value[..., 1])
        # for all other formats, use astropy mechanism
        return Time(value, scale=self.scale, format=self.format, copy=COPY_IF_NEEDED)

    @override
    def get_meta(self, colname):
        return {
            f"CTAFIELD_{colname}_TRANSFORM": "time",
            f"CTAFIELD_{colname}_TIME_FORMAT": self.format,
            f"CTAFIELD_{colname}_TIME_SCALE": self.scale,
        }


class QuantityColumnTransform(ColumnTransform):
    """A Column Transform that transforms quantities to their values in the given unit

    Parameters
    ----------
    unit : astropy.units.UnitBase or str
        Unit whose numeric values are stored and whose units are restored when reading.
    """

    def __init__(self, unit):
        self.unit = unit

    @override
    def __call__(self, value):
        return value.to_value(self.unit)

    @override
    def inverse(self, value):
        return Quantity(value, self.unit, copy=COPY_IF_NEEDED)

    @override
    def get_meta(self, colname):
        return {
            f"CTAFIELD_{colname}_TRANSFORM": "quantity",
            f"CTAFIELD_{colname}_UNIT": self.unit.to_string("vounit"),
        }


class FixedPointColumnTransform(ColumnTransform):
    """
    Apply a scale, offset and dtype conversion.

    Can be used to store values as fixed point by using an integer dtype
    and a scale that is a power of 10.

    The transforms reserves 3 integers to represent -inf, nan and inf.
    Underflowing values are converted to -inf and overflowing to inf.

    For unsigned target dtype:
    -inf: maxval - 2
    nan: maxval - 1
    inf: maxval

    For signed target dtype:
    -inf: minval
    nan: minval + 1
    inf: maxval

    When reading, these special values must not be interpreted as valid integer
    values but be transformed back into -inf, nan, inf respectively.

    This is a lossy transformation.

    Parameters
    ----------
    scale : float
        Multiplier applied before rounding the source values to integers.
    offset : int
        Integer offset added after rounding to shift the stored range.
    source_dtype : numpy.dtype or str
        Numeric dtype used for the source values and decoded results.
    target_dtype : numpy.dtype or str
        Integer storage dtype. Its range determines the reserved codes for nonfinite
        values.
    """

    def __init__(self, scale, offset, source_dtype, target_dtype):
        self.scale = scale
        self.offset = offset
        self.source_dtype = np.dtype(source_dtype)
        self.target_dtype = np.dtype(target_dtype)
        self.unsigned = self.target_dtype.kind == "u"

        iinfo = np.iinfo(self.target_dtype)

        # use three highest values for nan markers for unsigned case
        if self.unsigned:
            self.neginf = iinfo.max - 2
            self.nan = iinfo.max - 1
            self.posinf = iinfo.max

            # this leaves this inclusive range for the valid values
            self.minval = 0
            self.maxval = iinfo.max - 3
        else:
            self.neginf = iinfo.min
            self.nan = iinfo.min + 1
            self.posinf = iinfo.max

            # this leaves this inclusive range for the valid values
            self.minval = iinfo.min + 2
            self.maxval = iinfo.max - 1

    @override
    def __call__(self, value):
        is_scalar = np.array(value, copy=COPY_IF_NEEDED).shape == ()
        value = np.atleast_1d(value).astype(self.source_dtype, copy=COPY_IF_NEEDED)

        scaled = np.round(value * self.scale) + self.offset

        # convert under/overflow values to -inf/inf
        scaled[scaled > self.maxval] = np.inf
        scaled[scaled < self.minval] = -np.inf

        nans = np.isnan(scaled)
        pos_inf = np.isposinf(scaled)
        neg_inf = np.isneginf(scaled)

        result = scaled.astype(self.target_dtype)
        result[nans] = self.nan
        result[neg_inf] = self.neginf
        result[pos_inf] = self.posinf

        if is_scalar:
            return np.squeeze(result)

        return result

    @override
    def inverse(self, value):
        is_scalar = np.array(value, copy=COPY_IF_NEEDED).shape == ()
        value = np.atleast_1d(value)

        result = (value.astype(self.source_dtype) - self.offset) / self.scale
        result = np.atleast_1d(result)

        nans = value == self.nan
        pos_inf = value == self.posinf
        neg_inf = value == self.neginf

        result[nans] = np.nan
        result[neg_inf] = -np.inf
        result[pos_inf] = np.inf

        if is_scalar:
            return np.squeeze(result)

        return result

    @override
    def get_meta(self, colname: str):
        return {
            f"CTAFIELD_{colname}_TRANSFORM": "fixed_point",
            f"CTAFIELD_{colname}_TRANSFORM_SCALE": self.scale,
            f"CTAFIELD_{colname}_TRANSFORM_DTYPE": str(self.source_dtype),
            f"CTAFIELD_{colname}_TRANSFORM_OFFSET": self.offset,
            f"CTAFIELD_{colname}_NAN_VALUE": self.nan,
            f"CTAFIELD_{colname}_POSINF_VALUE": self.posinf,
            f"CTAFIELD_{colname}_NEGINF_VALUE": self.neginf,
        }


class EnumColumnTransform(ColumnTransform):
    """Store the value of an enum

    Parameters
    ----------
    enum : type[enum.Enum]
        Enum class whose member values are stored and used to reconstruct the members.
    """

    def __init__(self, enum):
        self.enum = enum

    @staticmethod
    @override
    def __call__(value):
        return value.value

    @override
    def inverse(self, value):
        return self.enum(value)

    @override
    def get_meta(self, colname):
        return {
            f"CTAFIELD_{colname}_TRANSFORM": "enum",
            f"CTAFIELD_{colname}_ENUM": self.enum,
        }


def encode_utf8_max_len(string, max_length):
    """Encode a string to utf-8 with max length in bytes

    This will not create invalid utf-8 data and thus the resulting
    bytes object might be shorter than ``max_length`` if max_length
    falls into a multi-byte codepoint.

    This might still create nonsensical output for cases like combining
    diacritics and emoji, but
    a) it will always successfully decode as utf-8
    b) we can probably live with not supporting all emojis
    when truncating strings in hdf5 data of ctapipe

    See https://stackoverflow.com/a/56724327/3838691

    Parameters
    ----------
    string : str
        Text to encode as UTF-8.
    max_length : int
        Maximum number of encoded bytes. Truncation preserves complete UTF-8 characters.
    """
    utf8 = string.encode("utf-8")

    if len(utf8) <= max_length:
        return utf8

    while (utf8[max_length] & 0b1100_0000) == 0b1000_0000:
        max_length -= 1

    return utf8[:max_length]


class StringTransform(ColumnTransform):
    """
    Encode strings as utf-8 bytes using a max_length

    Byte values are truncated after ``max_length``, this might result
    in invalid utf-8!. Trailing utf-8 bytes are ignored when inversing the
    transform.

    Should not be used anymore when tables starts supporting
    variable length strings in tables.

    Parameters
    ----------
    max_length : int
        Maximum width of the stored UTF-8 strings, in bytes.
    """

    def __init__(self, max_length):
        self.max_length = max_length
        self.dtype = f"S{max_length:d}"

    @override
    def __call__(self, value):
        if isinstance(value, str):
            return encode_utf8_max_len(value, self.max_length)

        return np.array(
            [encode_utf8_max_len(v, self.max_length) for v in value]
        ).astype(self.dtype)

    @override
    def inverse(self, value):
        if isinstance(value, bytes):
            return value.decode("utf-8")

        # astropy table columns somehow try to handle byte columns as strings
        # when iterating, this does not work here, convert to np.array
        value = np.array(value, copy=COPY_IF_NEEDED)
        return np.array([v.decode("utf-8") for v in value])

    @override
    def get_meta(self, colname):
        return {
            f"CTAFIELD_{colname}_TRANSFORM": "string",
            f"CTAFIELD_{colname}_MAXLEN": self.max_length,
        }


class TelListToMaskTransform(ColumnTransform):
    """convert variable-length list of tel_ids to a fixed-length mask

    Parameters
    ----------
    subarray : ctapipe.instrument.SubarrayDescription
        Subarray that defines the mask length and the order of telescope IDs along the
        mask axis.
    """

    def __init__(self, subarray: SubarrayDescription):
        self._forward = subarray.tel_ids_to_mask
        self._inverse = subarray.tel_mask_to_tel_ids

    @override
    def __call__(self, value):
        if value is None:
            return None

        return self._forward(value)

    @override
    def inverse(self, value):
        if value is None:
            return None
        return self._inverse(value)

    @override
    def get_meta(self, colname):
        return {f"CTAFIELD_{colname}_TRANSFORM": "tel_list_to_mask"}
