from abc import ABC, abstractmethod
from gzip import open as gzip_open
from astropy.io import fits
from pickle import load

__all__ = ['PickleSource', 'FITSSource']


class Source(ABC):
    def __init__(self, filename):
        self.infile = filename
        self.file_object = None

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass


class PickleSource(Source):
    """
    Reads a pickled file and yield containers
    """
    def __init__(self, filename):
        """
        Parameters
        ----------
        filename: str
             full path input file name
        """
        super().__init__(filename)
        self.file_object = gzip_open(filename, 'rb')

    def __next__(self):
        """
        Get next container in file

        Returns
        -------
        Next container in file

        Raises:
        ------
        EOFError: When end of file is reached without returning Container
        """
        try:
            container = load(self.file_object)
            return container
        except EOFError:
            raise StopIteration

    def close(self):
        """
        Close gzip file
        """
        self.file_object.close()

    def __iter__(self):
        """
        Iterate over all containers
        """
        return self


class FITSSource(Source):
    """
    Reads a FITS file and yield containers
    """
    def __init__(self, filename):
        """
        Parameters
        ----------
        filename: str
             full path input file name
        """
        super().__init__(filename)
        self.file_object = fits.open(filename)

    def __next__(self):
        pass

    def close(self):
        """
        Close FITS file
        """
        self.file_object.close()

    def __iter__(self):
        """
        Iterate over all containers
        """
        return self
