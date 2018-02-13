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

    def __next__(self):
        """
        Get next container in file

        Returns
        -------
        Next container in file

        Raises:
        ------
        StopIteration: When end of file is reached without returning Container
        """
        try:
            container = load(self.file_object)
            return container
        except EOFError:
            raise StopIteration

    def __iter__(self):
        """
        Iterate over all containers
        """
        return self

    def close(self):
        """
        Close gzip file
        """
        self.file_object.close()


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
        raise NotImplementedError("FITSSource not yet implemented")

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
