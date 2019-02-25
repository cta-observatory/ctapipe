"""
Handles seeking to a particular event in a
`ctapipe.io.eventfilereader.EventFileReader`
"""
from copy import deepcopy
from ctapipe.core import Component

__all__ = ['EventSeeker', ]


class EventSeeker(Component):
    """
    Provides the functionality to seek through a
    `ctapipe.io.eventfilereader.EventSource` to find a particular event.

    By default, this will loop through events from the start of the file
    (unless the requested event is the same as the previous requested event,
    or occurs later in the file). However if the
    `ctapipe.io.eventfilereader.EventSource` has defined a `__getitem__`
    method itself, then it will use that method, thereby taking advantage of
    the random event access some file formats provide.

    To create an instance of an EventSeeker you must provide it a sub-class of
    `ctapipe.io.eventfilereader.EventSource` (such as
    `ctapipe.io.hessiofilereader.HessioFileReader`), which will be used to
    loop through the file and provide the event container, filled with the
    event information using the methods defined in the event_source for that
    file format.

    To obtain a particular event in a hessio file:

    >>> from ctapipe.io.hessioeventsource import SimTelEventSource
    >>> event_source = SimTelEventSource(input_url="/path/to/file")
    >>> seeker = EventSeeker(event_source=event_source)
    >>> event = seeker[2]
    >>> print(event.count)

    To obtain a particular event in a hessio file from its event_id:

    >>> from ctapipe.io.hessioeventsource import SimTelEventSource
    >>> event_source = SimTelEventSource(input_url="/path/to/file")
    >>> seeker = EventSeeker(event_source=event_source)
    >>> event = seeker["101"]
    >>> print(event.count)

    **NOTE**: Event_index refers to the number associated to the event
    assigned by ctapipe (`event.count`), based on the order the events are
    read from the file.
    Whereas the event_id refers to the ID attatched to the event from the
    external source of the file (software or camera or CTA array).

    To obtain a slice of events in a hessio file:

    >>> from ctapipe.io import SimTelEventSource
    >>> event_source = SimTelEventSource(input_url="/path/to/file")
    >>> seeker = EventSeeker(event_source=event_source)
    >>> event_list = seeker[3:6]
    >>> print([event.count for event in event_list])

    To obtain a list of events in a hessio file:

    >>> from ctapipe.io import SimTelEventSource
    >>> event_source = SimTelEventSource(input_url="/path/to/file")
    >>> seeker = EventSeeker(event_source)
    >>> event_indicis = [2, 6, 8]
    >>> event_list = seeker[event_indicis]
    >>> print([event.count for event in event_list])
    """

    def __init__(self, reader, config=None, tool=None, **kwargs):
        """
        Class to handle generic input files. Enables obtaining the "source"
        generator, regardless of the type of file (either hessio or camera
        file).

        Parameters
        ----------
        reader : `ctapipe.io.eventfilereader.EventSource`
            A subclass of `ctapipe.io.eventfilereader.EventFileReader` that
            defines how the event container is filled for a particular file
            format
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
        super().__init__(config=config, tool=tool, **kwargs)

        if reader.is_stream:
            raise IOError("Reader is not compatible as input to the "
                          "event_source is a stream (seeking not possible)")

        self._reader = reader

        self._num_events = None
        self._source = self._reader.__iter__()
        self._current_event = None
        self._has_fast_seek = False  # By default seeking iterates through
        self._getevent_warn = True

    def _reset(self):
        """
        Recreate the generator so it starts from the beginning
        """
        self._source = self._reader.__iter__()
        self._current_event = None

    def __iter__(self):
        # Always reset generator when starting a new iteration
        self._reset()
        for event in self._source:
            self._current_event = event
            yield event

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

        # Handling of different input types (int, string, slice, list)
        current = None
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
            self._reset()

        # Check we are within max_events range
        max_events = self._reader.max_events
        if not use_event_id and max_events and item >= max_events:
            msg = ("Event index {} outside of specified max_events {}"
                   .format(item, max_events))
            raise IndexError(msg)

        try:
            if not use_event_id:
                event = self._reader._get_event_by_index(item)
            else:
                event = self._reader._get_event_by_id(item)
        except AttributeError:
            if self._getevent_warn:
                self.log.warning("Seeking to event by looping through "
                                 "events... (potentially long process)")
                self._getevent_warn = False
            if not use_event_id:
                event = self._get_event_by_index(item)
            else:
                event = self._get_event_by_id(item)

        self._current_event = event
        return deepcopy(event)

    def _get_event_by_index(self, index):
        """
        Method for extracting a particular event by looping through events
        until it finds the requested event index.
        If a file format allows random event access, then is can define its
        own `get_event_by_index` method in its
        `ctapipe.io.eventfilereader.EventSource` to allow this class to
        utilise that method instead.

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
                return event
        raise IndexError(f"Event index {index} not found in file")

    def _get_event_by_id(self, event_id):
        """
        Method for extracting a particular event by looping through events
        until it finds the requested event id.
        If a file format allows random event access, then is can define its
        own `get_event_by_id` method in its
        `ctapipe.io.eventfilereader.EventSource` to allow this class to
        utilise that method instead.

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
            if event.r0.event_id == event_id:
                return event
        raise IndexError(f"Event id {event_id} not found in file")

    def __len__(self):
        """
        Method for getting number of events in file. By default this is
        obtained by looping through the file and counting the events. If a
        file format has a more efficient method of supplying this information,
        the `ctapipe.io.eventfilereader.EventSource` for that file format
        can define its own `__len__` method, which this class will then
        use instead.

        Returns
        -------
        self._num_events : int
            Number of events in the file
        """
        # Only need to calculate once
        if not self._num_events:
            try:
                count = len(self._reader)
            except TypeError:
                self.log.warning("Obtaining length of file by looping through "
                                 "all events... (potentially long process)")
                count = 0
                for _ in self:
                    count += 1
            self._num_events = count
        return self._num_events
