"""
Serialize ctapipe containers to file
"""

from abc import ABC, abstractmethod
from gzip import open as gzip_open
from pickle import dump

import numpy as np
from astropy import log
from astropy.table import Table, Column
from traitlets import Unicode

from ctapipe.core import Container

__all__ = ['Serializer']


class Serializer:
    """
    Serializes ctapipe.core.Component, write it to a file thanks
    to its Writer object
    For some formats (i.e. pickle +gzip), read serialized components from
    a file

    Examples
    --------
    >>> writer = Serializer(filename='output.pickle', format='pickle', mode='w')
    >>> for container in input_containers:
    ...       writer.add_container(container.r0)
    >>> writer.close()

    or using the context manager syntax
    >>> with Serializer(filename='output.fits', format='fits', mode='w') as writer:
    >>> for container in input_containers:
    ...     writer.add_container(container.r0)
    """

    def __init__(self, filename, format='fits', mode='x'):

        """
        Parameters
        ----------
        filename: str
            full path name for i/o file
        format: str ('fits', 'img', 'pickle')
        mode: str ('write', 'read')
            : use this serializer as writer or reader
        mode: str
            'w'	open for writing, truncating the file first
            'x'	open for exclusive creation, failing if the file already exists
            'a'	open for writing, appending to the end of the file if it exists
        Raises
        ------
        NotImplementedError: when format is not implemented
        ValueError: when mode is not correct
        """
        self.filename = filename
        self.format = format
        self._stat = None  # TODO collect statistics about serialized contents
        if mode not in ('x', 'w', 'a'):
            raise ValueError('{} is not a valid write mode. Use x, w or a'.
                             format(mode))
        self._writer = None

        if self.format == 'fits':
            self._writer = TableWriter(outfile=filename, mode=mode,
                                       format=format)
        elif self.format == 'pickle':
            self._writer = GZipPickleWriter(outfile=filename, mode=mode)
        elif self.format == 'img':
            raise NotImplementedError('img serializer format is'
                                      ' not yet implemented')
        else:
            raise ValueError('You can serialize only on pickle, fits or img')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context related to this object.
        The parameters describe the exception that caused the context to be
        exited. If the context was exited without an exception,
        all three arguments will be None.
        If an exception is supplied, and the method wishes to suppress
        the exception (i.e., prevent it from being propagated),
        it should return a true value. Otherwise, the exception will be
        processed normally upon exit from this method.
        """
        self.close()

    def add_container(self, container):
        """
        Add a container to serializer
        """
        self._writer.add_container(container)


    def close(self):
        """
        Write data to disk
        """
        self._writer.close()


class Writer(ABC):

    def __init__(self, filename):
        self.outfile = filename

    @abstractmethod
    def add_container(self, container):
        pass

    @abstractmethod
    def close(self):
        pass


class GZipPickleWriter(Writer):
    """
    Serializes list of ctapipe.core.Components.
    Write Component to file
    """

    def __init__(self, outfile, mode='x'):
        """
        Parameters
        ----------
        outfile:  Unicode
            full path output file name
        mode: str
            'w'	open for writing, truncating the file first
            'x'	open for exclusive creation, failing if the file already exists
            'a'	open for writing, appending to the end of the file if it exists
        Raises
        ------
        FileNotFoundError: When the file cannot be opened
        FileExistsError: when infile exist and mode is x
        """
        super().__init__(outfile)
        mode += 'b'
        try:
            self.file_object = gzip_open(outfile, mode)
        except FileExistsError:
            raise FileExistsError('file exists: {} and mode is {}'.
                                  format(outfile, mode))

    def close(self):
        """
        close opened file
        Returns
        -------
        """
        self.file_object.close()

    def add_container(self, container):
        """
        Add a container to be serialized

        Raises
        ------
        TypeError: When container is not type of container
        """
        if not isinstance(container, Container):
            raise TypeError('Can write only Containers')
        dump(container, self.file_object)


# FITS Implementation

not_writeable_fields = ('tel', 'tels_with_data', 'calibration_parameters',
                        'pedestal_subtracted_adc', 'integration_window')


def is_writeable(key, out_format='fits'):
    """
    check if a key is writable

    Parameters
    ----------
    key: str
    out_format: 'fits' or ´ pickle'
        according to out_format a same key can be writable or not

    Returns
    -------
    True if key is writable according to the out_format

    Raises
    ------
    NameError: When out_format is not know
    """
    if out_format is 'fits':
        return not (key in not_writeable_fields)
    elif out_format is 'pickle':
        return True
    else:
        raise NameError('{} not implemented'.format(out_format))


def writeable_items(container):
    """
    # Strip off what we cannot write
    Parameters
    ----------
    container: ctapipe.core.Container

    Returns
    -------
    a dictionary with writable values only
    """

    d = dict(container.items())
    for k in not_writeable_fields:
        log.debug("Cannot write column {0}".format(k))
        d.pop(k, None)
    return d


def to_table(container):
    """
    Convert a `ctapipe.core.Container` to an `astropy.Table` with one row

    Parameters
    ----------
    container: ctapipe.core.Container

    Returns
    -------
    Table: astropy.Table
    """
    names = list()
    columns = list()
    for k, v in writeable_items(container).items():

        v_arr = np.array(v)
        v_arr = v_arr.reshape((1,) + v_arr.shape)
        log.debug("Creating column for item '{0}' of shape {1}".
                  format(k, v_arr.shape))
        names.append(k)
        columns.append(Column(v_arr))

    return Table(data=columns,  # dtypes are inferred by columns
                 names=names,
                 meta=container.meta)


class TableWriter(Writer):
    """
    Fits table writer
    """

    def __init__(self, outfile, format='fits', mode='w'):
        """
        Parameters
        ----------
        outfile: str
            output file name
        format: str
            'fits' or 'img'
        mode: str
            'w'	open for writing, truncating the file first
            'x'	open for exclusive creation, failing if the file already exists
        Raises
        ------
        NotImplementedError: when mode is correct but not yet implemented
        ValueError: when mode is not correct
        """
        super().__init__(outfile)
        self.table = Table()
        self._created_table = False
        self.format = format
        self.outfile = outfile
        if mode == 'w':
            self.overwrite = True
        elif mode == 'x':
            self.overwrite = False
        elif mode == 'a':
            raise NotImplementedError('a is a valid write mode,'
                                      ' but not yet implemented')
        else:
            raise ValueError('{} is not a valid write mode. Use x, w or a'.
                             format(mode))

    def _setup_table(self, container):
        """
        Create Fits table and HDU

        Parameters
        ----------
        container: ctapipe.core.Container
        """
        # Create Table from Container
        self.table = to_table(container)

        # Write HDU name
        if self.format == "fits":
            self.table.meta["EXTNAME"] = type(container).__name__
        self._created_table = True

    def add_container(self, container):
        """
        Add a container as a table row
        Parameters
        ----------
        container: ctapipe.core.Container

        Raises
        ------
        TypeError: When add another type than Container
        """
        if not isinstance(container, Container):
            raise TypeError("Can write only Containers")

        if not self._created_table:
            self._setup_table(container)
        else:
            self.table.add_row(writeable_items(container))

    def close(self, **kwargs):
        """
        Write Fits table to file
        Parameters
        ----------
        kwargs to be passed to `astropy.Table.write method`

        Returns
        -------
        Fits Table
        """
        # Write table using astropy.table write method
        self.table.write(output=self.outfile, format=self.format,
                         overwrite=self.overwrite, **kwargs)
        return self.table
