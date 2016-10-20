"""
Serialize ctapipe containers to file
"""

from astropy.table import Table, Column
from ctapipe.core import Container
from ctapipe.core import Component
from abc import ABC, abstractmethod
from astropy import log
from pickle import load
from pickle import dump
from pickle import PickleError
from traitlets import Unicode
import numpy as np
import gzip

__all__ = ['FitsSerializer', 'PickleSerializer']
log.setLevel('INFO')


# PICKLE GZIP Implementation

class Serializer(ABC, Component):
    """
    Base class for context manager to save Containers to file
    """
    def __init__(self, file,  overwrite=False):
        """
        Parameters
        ----------
        file:  Unicode
            input or output full path file name
        overwrite: Bool
            overwrite outfile file if it already exists
        """
        self.file = file
        self.overwrite = overwrite
        super().__init__()

    def __enter__(self):
        log.debug("Serializing on {0}".format(self.file))
        return self

    def __exit__(self, *args):
        pass

    @abstractmethod
    def add_container(self, container):
        """
        Add a container to serializer
        """
        pass

    @abstractmethod
    def write(self):
        """
        Write all containers to outfile
        """
        pass

    @abstractmethod
    def __iter__(self):
        """
        Iterate over all containers
        """
        pass


class PickleSerializer(Serializer):
    """
    Serializes list of ctapipe.core.Components.
    Reads list of Components from file
    Write list of Components to file
    """
    file = Unicode('file name', help='serializer file name').tag(
        config=True)

    def __init__(self, file=None, overwrite=False):
        """
       Parameters
       ----------
       file:  Unicode
          input/output full path file name
       overwrite: Bool
           overwrite outfile file if it already exists
        """
        if file:
            self.file = file
        super().__init__(self.file, overwrite)
        self.containers = []

    def add_container(self, container):
        """
        Add a container to serializer
        Raise
        -----
        TypeError : When container is not type of container
        """
        if not isinstance(container, Container):
            raise TypeError('Can write only Containers')
        self.containers.append(container)

    def write(self):
        """
        Write all containers to outfile
        Return : str
             output file with pickle.gzip extension added if it is not already
              presents.
        Raise
        -----
        PickleError: When containers list cannot be write to outfile
        PermissionError:
        FileNotFoundError:
        """
        try:
            file_split = self.file.split('.')
            if file_split[-1] != 'gzip' or file_split[-2] != 'pickle':
                output_filename = self.file+'.pickle.gzip'
            else:
                output_filename = self.file
            with gzip.open(output_filename, 'wb') as f_out:
                dump(self.containers, f_out)
            return output_filename
        except (PickleError, PermissionError, FileNotFoundError):
            raise

    def __iter__(self):
        """
        Iterate over all containers
        Return an iterator object
        """
        try:
            with gzip.open(self.file, 'rb') as f:
                self.containers = load(f)
                for container in self.containers:
                    yield container
        except PickleError:
            raise

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
        log.debug("Creating column for item '{0}' of shape {1}".format(k, v_arr.shape))
        names.append(k)
        columns.append(Column(v_arr))
    return names, columns


class TableWriter:
    """
    Fits table writer
    """
    def __init__(self, outfile, format, overwrite):
        """
        Parameters
        ----------
        outfile: str
            output file name
        format: str
            'fits' or 'img'
        overwrite: bool
        """
        self.table = Table()
        self._created_table = False
        self.format = format
        self.outfile = outfile
        self.overwrite = overwrite

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
        self.table.write(output=self.outfile, format=self.format, overwrite=self.overwrite, **kwargs)
        return self.table


class FitsSerializer(Serializer):
    """
    Serializes list of ctapipe.core.Components.
    Reads list of Components from fits file
    Write list of Components to fits file

    TO DO : Implement reader as __iter__
    """
    file = Unicode('file name', help='serializer file name').tag(
        config=True)

    def __init__(self, outfile=None, format='fits', overwrite=False):
        """
        Parameters
        ----------
        outfile: Unicode
            output file name
        format: str
            'fits' or 'img'
        overwrite: bool
        """
        self.outfile = outfile
        if outfile:
            self.file = outfile
        self._writer = TableWriter(outfile=self.file, format=format,
                                   overwrite=overwrite)
        super().__init__(self.file, overwrite)

    def __enter__(self):
        """
        Executed when “with” statement is executed
        Returns
        -------
        self
        """
        log.debug("Serializing on {0}".format(self.outfile))
        return self

    def __exit__(self, *args):
        """
        Executed at the end of  “with” statement execution
        Write
        Parameters
        ----------
        args
        """
        pass

    def __iter__(self):
        """
        Return an iterator object
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def add_container(self, container):
        """
        Add container fo Fits
        Parameters
        ----------
        container: ctapipe.core.Container
        """
        self._writer.add_container(container)

    def write_source(self, source):
        raise NotImplementedError

    def write(self):
        """
        Write Fits Table to Fits file
        """
        self._writer.write()

