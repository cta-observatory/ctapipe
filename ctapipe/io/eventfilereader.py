"""
Handles reading of different event/waveform containing files
"""
from abc import abstractmethod
from os.path import exists
from traitlets import Unicode, Int, CaselessStrEnum
from copy import deepcopy
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
    configuration (containing the input_path) and the
    `ctapipe.core.tool.Tool`. Therefore from inside a Tool you would do:

    >>> reader = EventFileReader(self.config, self)

    An example of how to use `ctapipe.core.tool.Tool` and
    `ctapipe.io.eventfilereader.EventFileReaderFactory` can be found in
    ctapipe/examples/calibration_pipeline.py.

    However if you are not inside a Tool, you can still create an instance and
    supply an input_path via:

    >>> reader = EventFileReader(None, None, input_path=path)

    To loop through the events in a file:

    >>> reader = EventFileReader(None, None, input_path=path)
    >>> for event in reader:
    >>>    print(event.count)

    **NOTE**: Every time a new loop is started through the reader, it restarts
    from the first event.

    To obtain a particular event in a file:

    >>> reader = EventFileReader(None, None, input_path=path)
    >>> event = reader[event_index]
    >>> print(event.count)

    To obtain a particular event in a file from its event_id:

    >>> reader = EventFileReader(None, None, input_path=path)
    >>> event = reader["event_id"]
    >>> print(event.count)

    **NOTE**: Event_index refers to the number associated to the event
    assigned by ctapipe (`event.count`), based on the order the events are
    read from the file.
    Whereas the event_id refers to the ID attatched to the event from the
    external source of the file (software or camera or CTA array).

    To obtain a slice of events in a file:

    >>> reader = EventFileReader(None, None, input_path=path)
    >>> event_list = reader[3:6]
    >>> print([event.count for event in event_list])

    To obtain a list of events in a file:

    >>> reader = EventFileReader(None, None, input_path=path)
    >>> event_indicis = [2, 6, 8]
    >>> event_list = reader[event_indicis]
    >>> print([event.count for event in event_list])

    Alternatively one can use EventFileReader in a `with` statement to ensure
    the correct cleanups are performed when you are finished with the reader:

    >>> with EventFileReader(None, None, input_path=path) as reader:
    >>>    for event in reader:
    >>>       print(event.count)

    Attributes
    ----------
    input_path : str
        Path to the input event file.
    max_events : int
        Maximum number of events to loop through in generator
    """

    input_path = Unicode('', allow_none=False,
                         help='Path to the input file containing '
                              'events.').tag(config=True)
    max_events = Int(None, allow_none=True,
                     help='Maximum number of events that will be read from'
                          'the file').tag(config=True)

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

        self._num_events = None
        self._metadata = dict(is_simulation=False)

        # if self.input_path is None:
        #     raise ValueError("Please specify an input_path for event file")
        if not exists(self.input_path):
            raise FileNotFoundError("file path does not exist: '{}'"
                                    .format(self.input_path))
        self.log.info("INPUT PATH = {}".format(self.input_path))

        if self.max_events:
            self.log.info("Max events being read = {}".format(self.max_events))

        Provenance().add_input_file(self.input_path, role='dl0.sub.evt')

        self._source = self._generator()
        self._current_event = None
        self._has_fast_seek = False  # By default seeking iterates through
        self._getevent_warn = True

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
    def metadata(self):
        """
        A dictionary containing the metadata of the file. This could include:
        * is_simulation (bool indicating if the file contains simulated events)
        * Telescope:Camera names (list if file contains multiple)
        * Information in the file header
        * Observation ID

        Returns
        -------
        dict
        """
        return self._metadata

    @property
    def is_stream(self):
        """
        Bool indicating if input is a stream. If it is then `__getitem__` and
        `__len__` are disabled.

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

    def reset(self):
        """
        Recreate the generator so it starts from the beginning
        """
        self._source = self._generator()
        self._current_event = None

    def __iter__(self):
        # Always reset generator when starting a new iteration
        self.reset()
        for event in self._source:
            if self.max_events and event.count >= self.max_events:
                break
            self._current_event = event
            yield event

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __getitem__(self, item):
        """
        Obtain a particular event

        Parameters
        ----------
        item : int or str
            If `item` is an int, then this is the event_index for the event
            obtained. If `item` is a str, then this is the event_id for the
            event obtained.

        Returns
        -------
        event : ctapipe.io.container
            The event container filled with the requested event's information

        """
        if self.is_stream:
            raise IOError("Input is a stream, __getitem__ is disabled")

        current = None

        if not self._has_fast_seek and self._getevent_warn:
            self.log.warning("Seeking to event... (potentially long process)")
            self._getevent_warn = False

        # Handling of different input types (int, string, slice, list)
        use_event_id = False
        if isinstance(item, int):
            if self._current_event:
                current = self._current_event.count
            if item < 0:
                item = len(self) + item
                if item < 0 or item >= len(self):
                    msg = ("Event index {} out of range [0, {}]"
                        .format(item, len(self)))
                    raise IndexError(msg)
        elif isinstance(item, str):
            item = int(item)
            use_event_id = True
            if self._current_event:
                current = self._current_event.r0.event_id
        elif isinstance(item, slice):
            it = range(item.start or 0, item.stop or len(self), item.step or 1)
            events = [self[i] for i in it]
            return events
        elif isinstance(item, list):
            events = [self[i] for i in item]
            return events
        else:
            raise TypeError("{} indexing is not supported".format(type(item)))

        # Return a copy of the current event if we have already reached it
        if current is not None and item == current:
            return deepcopy(self._current_event)

        # If requested event is less than the current event position: reset
        if current is not None and item < current:
            self.reset()

        # Check we are within max_events range
        if not use_event_id and self.max_events and item >= self.max_events:
            msg = ("Event index {} outside of specified max_events {}"
                .format(item, self.max_events))
            raise IndexError(msg)

        if not use_event_id:
            return self._get_event_by_index(item)
        else:
            return self._get_event_by_id(item)

    def _get_event_by_index(self, index):
        """
        Method for extracting a particular event for this file format by
        event index.
        If a file format allows random event access, this function can be
        overrided for a more efficient method.

        Parameters
        ----------
        index : int
            The event_index to seek.

        Returns
        -------
        event : ctapipe.io.container
            The event container filled with the requested event's information

        """
        for event in self._source:
            if event.count == index:
                self._current_event = event
                return deepcopy(event)
        raise IndexError("Event index {} not found in file".format(index))

    def _get_event_by_id(self, event_id):
        """
        Method for extracting a particular event for this file format by
        event id.
        If a file format allows random event access, this function can be
        overrided for a more efficient method.

        Parameters
        ----------
        event_id : int
            The event_id to seek.

        Returns
        -------
        event : ctapipe.io.container
            The event container filled with the requested event's information

        """
        for event in self:  # Event Ids may not be in order
            if self.max_events and event.count >= self.max_events:
                break
            if event.r0.event_id == event_id:
                self._current_event = event
                return deepcopy(event)
        raise IndexError("Event id {} not found in file".format(event_id))

    def __len__(self):
        if self.is_stream:
            raise IOError("Input is a stream, __len__ is disabled")

        # Only need to calculate once
        if not self._num_events:
            self.reset()
            self.log.warning("Obtaining length of file... "
                             "(potentially long process)")
            count = 0
            for _ in self:
                if self.max_events and count >= self.max_events:
                    break
                count += 1
            self._num_events = count
        return self._num_events


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

    reader = CaselessStrEnum(subclass_names, None, allow_none=True,
                             help='Event file reader to use. If None then '
                                  'a reader will be chosen based on file '
                                  'extension').tag(config=True)

    # Product classes traits
    # Would be nice to have these automatically set...!
    input_path = Unicode(get_dataset('gamma_test.simtel.gz'), allow_none=True,
                         help='Path to the input file containing '
                              'events.').tag(config=True)
    max_events = Int(None, allow_none=True,
                     help='Maximum number of events that will be read from'
                          'the file').tag(config=True)

    def get_factory_name(self):
        return self.__class__.__name__

    def get_product_name(self):
        if self.reader is not None:
            return self.reader
        else:
            if self.input_path is None:
                raise ValueError("Please specify an input_path for event file")
            try:
                for subclass in self.subclasses:
                    if subclass.is_compatible(self.input_path):
                        return subclass.__name__
                raise ValueError
            except ValueError:
                self.log.exception("Cannot find compatible EventFileReader "
                                   "for: {}".format(self.input_path))
                raise
