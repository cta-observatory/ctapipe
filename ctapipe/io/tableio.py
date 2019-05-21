import re
from abc import ABCMeta, abstractmethod
from collections import defaultdict

from ctapipe.core import Component, Container

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
        self._exclusions[table_name].append(re.compile(pattern))

    def _is_column_excluded(self, table_name, col_name):
        for pattern in self._exclusions[table_name]:
            if pattern.match(col_name):
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
        self._transforms[table_name][col_name] = transform
        self.log.debug(
            "Added transform: {}/{} -> {}".format(table_name, col_name, transform)
        )

    @abstractmethod
    def write(self, table_name, containers):
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
        container: `ctapipe.core.Container`
            container to write
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
        self._cols_to_read = defaultdict(list)
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
            value = tr(value)
        return value

    @abstractmethod
    def read(self, table_name: str, container: Container):
        """
        Returns a generator that reads the next row from the table into the
        given container.  The generator returns the same container. Note that
        no containers are copied, the data are overwritten inside.

        Parameters
        ----------
        table_name: str
            name of table to read from
        container : ctapipe.core.Container
            Container instance to fill
        """
        pass

    @abstractmethod
    def open(self, filename, **kwargs):
        pass

    @abstractmethod
    def close(self):
        pass
