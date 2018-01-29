"""
Handles reading of different event/waveform containing files
"""
from abc import abstractmethod
from os.path import exists
from traitlets import Unicode, Int, CaselessStrEnum
from ctapipe.core import Component, Factory
from ctapipe.utils import get_dataset
from ctapipe.core import Provenance


class EventFileReader(Component):
    """
    Parent class for EventFileReaders of different sources.

    A new EventFileReader should be created for each type of event file read
    into ctapipe, e.g. sim_telarray files are read by the `HessioFileReader`.

    EventFileReader provides a common high-level interface for accessing event
    information from different data sources (simulation or different camera
    file formats). Creating an EventFileReader for a new
    file format ensures that data can be accessed in a common way,
    irregardless of the file format.

    EventFileReader itself is an abstract class. To use an EventFileReader you
    must use a subclass that is relevant for the file format you
    are reading (for example you must use
    `ctapipe.io.hessiofilereader.HessioFileReader` to read a hessio format
    file). Alternatively you can use
    `ctapipe.io.eventfilereader.EventFileReaderFactory` to automatically
    select the correct EventFileReader subclass for the file format you wish
    to read.

    To create an instance of an EventFileReader you must pass the traitlet
    configuration (containing the input_url) and the
    `ctapipe.core.tool.Tool`. Therefore from inside a Tool you would do:

    >>> reader = EventFileReader(self.config, self)

    An example of how to use `ctapipe.core.tool.Tool` and
    `ctapipe.io.eventfilereader.EventFileReaderFactory` can be found in
    ctapipe/examples/calibration_pipeline.py.

    However if you are not inside a Tool, you can still create an instance and
    supply an input_url via:

    >>> reader = EventFileReader(None, None, input_url="/path/to/file")

    To loop through the events in a file:

    >>> reader = EventFileReader(None, None, input_url="/path/to/file")
    >>> for event in reader:
    >>>    print(event.count)

    **NOTE**: Every time a new loop is started through the reader, it restarts
    from the first event.

    Alternatively one can use EventFileReader in a `with` statement to ensure
    the correct cleanups are performed when you are finished with the reader:

    >>> with EventFileReader(None, None, input_url="/path/to/file") as reader:
    >>>    for event in reader:
    >>>       print(event.count)

    **NOTE**: The "event" that is returned from the generator is a pointer.
    Any operation that progresses that instance of the generator further will
    change the data pointed to by "event". If you wish to ensure a particular
    event is kept, you should perform a `event_copy = copy.deepcopy(event)`.


    Attributes
    ----------
    input_url : str
        Path to the input event file.
    max_events : int
        Maximum number of events to loop through in generator
    metadata : dict
        A dictionary containing the metadata of the file. This could include:
        * is_simulation (bool indicating if the file contains simulated events)
        * Telescope:Camera names (list if file contains multiple)
        * Information in the file header
        * Observation ID
    """

    input_url = Unicode(
        '',
        help='Path to the input file containing events.'
    ).tag(config=True)
    max_events = Int(
        None,
        allow_none=True,
        help='Maximum number of events that will be read from the file'
    ).tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        Class to handle generic input files. Enables obtaining the "source"
        generator, regardless of the type of file (either hessio or camera
        file).

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, parent=tool, **kwargs)

        self.metadata = dict(is_simulation=False)

        if not exists(self.input_url):
            raise FileNotFoundError("file path does not exist: '{}'"
                                    .format(self.input_url))
        self.log.info("INPUT PATH = {}".format(self.input_url))

        if self.max_events:
            self.log.info("Max events being read = {}".format(self.max_events))

        Provenance().add_input_file(self.input_url, role='dl0.sub.evt')

    @staticmethod
    @abstractmethod
    def is_compatible(file_path):
        """
        Abstract method to be defined in child class.

        Perform a set of checks to see if the input file is compatible
        with this file reader.

        Parameters
        ----------
        file_path : str
            File path to the event file.

        Returns
        -------
        compatible : bool
            True if file is compatible, False if it is incompatible
        """

    @property
    def is_stream(self):
        """
        Bool indicating if input is a stream. If it is then it is incompatible
        with `ctapipe.io.eventseeker.EventSeeker`.

        TODO: Define a method to detect if it is a stream

        Returns
        -------
        bool
            If True, then input is a stream.
        """
        return False

    @abstractmethod
    def _generator(self):
        """
        Abstract method to be defined in child class.

        Generator where the filling of the `ctapipe.io.containers` occurs.

        Returns
        -------
        generator
        """

    def __iter__(self):
        """
        Generator that iterates through `_generator`, but keeps track of
        `self.max_events`.

        Returns
        -------
        generator
        """
        for event in self._generator():
            if self.max_events and event.count >= self.max_events:
                break
            yield event

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# EventFileReader imports so that EventFileReaderFactory can see them
import ctapipe.io.hessiofilereader


class EventFileReaderFactory(Factory):
    """
    The `EventFileReader` `ctapipe.core.factory.Factory`. This
    `ctapipe.core.factory.Factory` allows the correct
    `EventFileReader` to be obtained for the event file being read.

    This factory tests each EventFileReader by calling
    `EventFileReader.check_file_compatibility` to see which `EventFileReader`
    is compatible with the file.

    Using `EventFileReaderFactory` in a script allows it to be compatible with
    any file format that has an `EventFileReader` defined.

    To use within a `ctapipe.core.tool.Tool`:

    >>> reader = EventFileReaderFactory.produce(config=self.config, tool=self)

    Parameters
    ----------
    config : traitlets.loader.Config
        Configuration specified by config file or cmdline arguments.
        Used to set traitlet values.
        Set to None if no configuration to pass.
    tool : ctapipe.core.Tool or None
        Tool executable that is calling this component.
        Passes the correct logger to the component.
        Set to None if no Tool to pass.
    kwargs

    Attributes
    ----------
    reader : traitlets.CaselessStrEnum
        A string with the `EventFileReader.__name__` of the reader you want to
        use. If left blank, `EventFileReader.check_file_compatibility` will be
        used to find a compatible reader.
    """
    description = "Obtain EventFileReader based on file type"

    subclasses = Factory.child_subclasses(EventFileReader)
    subclass_names = [c.__name__ for c in subclasses]

    reader = CaselessStrEnum(
        subclass_names,
        None,
        allow_none=True,
        help='Event file reader to use. If None then a reader will be chosen '
             'based on file extension'
    ).tag(config=True)

    # Product classes traits
    # Would be nice to have these automatically set...!
    input_url = Unicode(
        get_dataset('gamma_test.simtel.gz'),
        help='Path to the input file containing events.'
    ).tag(config=True)
    max_events = Int(
        None,
        allow_none=True,
        help='Maximum number of events that will be read from the file'
    ).tag(config=True)

    def get_product_name(self):
        if self.reader is not None:
            return self.reader
        else:
            if self.input_url is None:
                raise ValueError("Please specify an input_url for event file")
            try:
                for subclass in self.subclasses:
                    if subclass.is_compatible(self.input_url):
                        return subclass.__name__
                raise ValueError
            except ValueError:
                self.log.exception("Cannot find compatible EventFileReader "
                                   "for: {}".format(self.input_url))
                raise
