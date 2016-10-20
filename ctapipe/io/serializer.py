"""

Serialize ctapipe containers to file
"""

from astropy.table import Table, Column
from ctapipe.core import Container
from abc import ABC, abstractmethod
from astropy import log
from pickle import load
from pickle import dump
from pickle import PickleError
import gzip

__all__ = ["Serializer"]
log.setLevel('INFO')

not_writeable_fields = ('tel', 'tels_with_data', 'calibration_parameters',
                        'pedestal_subtracted_adc', 'integration_window')


class Serializer(ABC):
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
    Serializes list of Components.
    Reads list of Components from file
    Write list of Components to file

    TO DO gzip
    """
    def __init__(self, file, overwrite=False):
        """
           Parameters
           ----------
           file:  Unicode
              input/output full path file name
           overwrite: Bool
               overwrite outfile file if it already exists
        """
        super().__init__(file, overwrite)
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
        """
        try:
            with gzip.open(self.file, 'rb') as f:
                self.containers = load(f)
                for container in self.containers:
                    yield container
        except PickleError:
            raise


def is_writeable(key, out_format='fits'):

    if out_format is 'fits':
        return not (key in not_writeable_fields)
    elif out_format is 'pickle':
        return True
    else:
         raise NotImplementedError('Format {} not implemented'.format(out_format))


def writeable_items(container):
    # Strip off what we cannot write
    d = dict(container.items())
    for k in not_writeable_fields:
        log.debug("Cannot write column {0}".format(k))
        d.pop(k, None)
    return d


def to_table(container):

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
    def __init__(self, outfile, out_format, overwrite):
        self.table = Table()
        self._created_table = False
        self.out_format = out_format
        self.outfile = outfile
        self.overwrite = overwrite

    def _setup_table(self, container):
        # Create Table from Container

        names, columns = to_table(container)
        self.table = Table(data=columns,  # dtypes are inferred by columns
                           names=names,
                           meta=container.meta.as_dict())
        # Write HDU name
        if self.out_format == "fits":
            self.table.meta["EXTNAME"] = container._name
        self._created_table = True

    def write(self, container):
        if not isinstance(container, Container):
            log.error("Can write only Containers")
            return
        if not self._created_table:
            self._setup_table(container)
        else:
            self.table.add_row(writeable_items(container))

    def save(self, **kwargs):
        # Write table using astropy.table write method
        self.table.write(output=self.outfile, format=self.out_format, overwrite=self.overwrite, **kwargs)
        return self.table
