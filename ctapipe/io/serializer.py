"""
Serialize ctapipe containers to file
"""

from astropy.table import Table, Column
from ctapipe.core import Container
from abc import ABC, abstractmethod
from astropy import log
from pickle import dump
from pickle import load
from traitlets import Unicode
import numpy as np
from gzip import open as gzip_open

__all__ = ['Serializer']


class Serializer:
    """
    Serializes ctapipe.core.Component, write it to a file thanks
    to its Writer object
    For some formats (i.e. pickle +gzip), read serialized components from
    a file
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
            'r'	open for reading
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

        if mode not in ['r', 'x', 'w', 'a']:
            raise ValueError('{} is not a valid write mode. Use x, w or a'.
                             format(mode))
        self._writer = None
        self._reader = None

        if self.format == 'fits':
            self._writer = TableWriter(outfile=filename, mode=mode,
                                       format=format)
        elif self.format == 'pickle':
            if mode == 'r':
                self._reader = GZipPickleReader(infile=filename)
            else:
                self._writer = GZipPickleWriter(outfile=filename,
                                                mode=mode)
        elif self.format == 'img':
            raise NotImplementedError('img serializer format is'
                                      ' not yet implemented')

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

    def close(self):
        """
        Close reader or writer
        """
        if self._writer:
            self._writer.write()
        elif self._reader:
            self._reader.close()

    def add_container(self, container):
        """
        Add a container to serializer
         Raises
        ------
        RuntimeError: When Serializer is used as Writer
        """
        if not self._writer:
            raise RuntimeError('This serializer instance is a reader')
        self._writer.add_container(container)

    def get_next_container(self):
        """
        Returns
        -------
        The next container in file
        Raises
        ------
        EOFError:  When end of file is reached without returning Container
        RuntimeError: When Serializer is used as writer
        """
        if not self._reader:
            raise RuntimeError('This serializer instance is a writer')
        return self._reader.get_next_container()

    def write(self):
        """
        Write data to disk OR close it
        Raises
        ------
        RuntimeError: When Serializer is used as reader
        """
        if not self._writer:
            raise RuntimeError('This serializer instance is a reader')
        self._writer.write()

    def __iter__(self):
        """
        Yields
        ------
        A container
        Raises
        ------
        RuntimeError: when this Serializer instance is a writer
        """
        if self._reader:
            for container in self._reader:
                yield container
            raise StopIteration
        else:
            raise RuntimeError('This Serializer instance is a writer')


class Writer(ABC):
    def __init__(self, outfile):
        self.outfile = outfile

    @abstractmethod
    def add_container(self, container):
        pass

    @abstractmethod
    def write(self):
        pass


class Reader(ABC):
    def __init__(self, infile):
        self.infile = infile

    @abstractmethod
    def get_next_container(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass


class GZipPickleReader(Reader):
    def __init__(self, infile):
        """
        Parameters
        ----------
        infile: str
             full path input file name
        """
        super().__init__(infile)
        self.file_object = gzip_open(infile, 'rb')

    def get_next_container(self):
        """
        Returns
        -------
        Next container in file
        Raises:
          EOFError: When end of file is reached without returning Container
        """
        return load(self.file_object)

    def close(self):
        """
        Close gzip file
        """
        self.file_object.close()

    def __iter__(self):
        """
        Iterate over all containers
        Return an iterator object
        Raises
        ------
        StopIteration: when all containers have been read
        """
        try:
            while True:
                container = load(self.file_object)
                yield container

        except EOFError:
            raise StopIteration


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

    def write(self):
        """
        close opened file
        Returns
        -------
        """
        self.file_object.close()

    def add_container(self, container):
        """
        Add a container to serializer
        Raises
        ------
        TypeError : When container is not type of container
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
    Return tuple of 2 lists:  names and columns from container
    Parameters
    ----------
    container: ctapipe.core.Container
    Returns
    -------
    tuple of 2 lists:  names and columns from container
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
    return names, columns


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
        names, columns = to_table(container)
        self.table = Table(data=columns,  # dtypes are inferred by columns
                           names=names,
                           meta=container.meta.as_dict())
        # Write HDU name
        if self.format == "fits":
            self.table.meta["EXTNAME"] = container._name
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

    def write(self, **kwargs):
        """
        Write Fits table to file
        Parameters
        ----------
        kwargs

        Returns
        -------
        Fits Table
        """
        # Write table using astropy.table write method
        self.table.write(output=self.outfile, format=self.format,
                         overwrite=self.overwrite, **kwargs)
        return self.table
