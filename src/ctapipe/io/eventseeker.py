"""
Handles seeking to a particular event in a `ctapipe.io.EventSource`
"""
from copy import deepcopy

from ctapipe.core import Component

__all__ = ["EventSeeker"]


class EventSeeker(Component):
    """
    Provides the functionality to seek through a
    `~ctapipe.io.EventSource` to find a particular event.

    By default, this will loop through events from the start of the file
    (unless the requested event is the same as the previous requested event,
    or occurs later in the file). However if the
    `ctapipe.io.EventSource` has defined a ``__getitem__``
    method itself, then it will use that method, thereby taking advantage of
    the random event access some file formats provide.

    To create an instance of an EventSeeker you must provide it a sub-class of
    `~ctapipe.io.EventSource` (such as `ctapipe.io.SimTelEventSource`),
    which will be used to loop through the file and provide the event container,
    filled with the event information using the methods defined in the
    event_source for that file format.

    To obtain a particular event in a simtel file:

    >>> from ctapipe.io import SimTelEventSource
    >>> event_source = SimTelEventSource(input_url="dataset://gamma_test_large.simtel.gz", focal_length_choice="EQUIVALENT")
    >>> seeker = EventSeeker(event_source=event_source)
    >>> event = seeker.get_event_index(2)
    >>> print(event.count)
    2

    To obtain a particular event in a simtel file from its event_id:

    >>> from ctapipe.io import SimTelEventSource
    >>> event_source = SimTelEventSource(input_url="dataset://gamma_test_large.simtel.gz", back_seekable=True, focal_length_choice="EQUIVALENT")
    >>> seeker = EventSeeker(event_source=event_source)
    >>> event = seeker.get_event_id(31007)
    >>> print(event.count)
    1

    **NOTE**: Event_index refers to the number associated to the event
    assigned by ctapipe (``event.count``), based on the order the events are
    read from the file.
    Whereas the event_id refers to the ID attached to the event from the
    external source of the file (software or camera or CTA array).
    """

    def __init__(self, event_source, config=None, parent=None, **kwargs):
        """
        Class to handle generic input files. Enables obtaining the "source"
        generator, regardless of the type of file (either hessio or camera
        file).

        Parameters
        ----------
        event_source : `ctapipe.io.eventsource.EventSource`
            A subclass of `ctapipe.io.eventsource.EventSource` that
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
        super().__init__(config=config, parent=parent, **kwargs)

        self._event_source = event_source

        self._n_events = None
        self._source = self._event_source.__iter__()
        self._current_event = None
        self._has_fast_seek = False  # By default seeking iterates through
        self._getevent_warn = True

    def _reset(self):
        """
        Recreate the generator so it starts from the beginning
        """
        if self._event_source.is_stream:
            raise OSError("Back-seeking is not possible for event source")
        self._source = self._event_source.__iter__()
        self._current_event = None

    def __iter__(self):
        # Always reset generator when starting a new iteration
        self._reset()
        for event in self._source:
            self._current_event = event
            yield event

    def get_event_index(self, event_index):
        """
        Obtain the event via its event index

        Parameters
        ----------
        event_index : int
            The event_index to seek.

        Returns
        -------
        event : ctapipe.io.container
            The event container filled with the requested event's information
        """
        if self._current_event and event_index == self._current_event.count:
            return deepcopy(self._current_event)

        # Check we are within max_events range
        max_events = self._event_source.max_events
        if max_events and event_index >= max_events:
            msg = f"Event index {event_index} is beyond max_events {max_events}"
            raise IndexError(msg)

        try:
            event = self._event_source._get_event_by_index(event_index)
        except AttributeError:
            event = self._get_event_by_index(event_index)

        self._current_event = event
        return deepcopy(event)

    def get_event_id(self, event_id):
        """
        Obtain the event via its event id

        Parameters
        ----------
        event_id : int
            The event_id to seek.

        Returns
        -------
        event : ctapipe.io.container
            The event container filled with the requested event's information

        """
        if self._current_event and event_id == self._current_event.index.event_id:
            return deepcopy(self._current_event)

        try:
            event = self._event_source._get_event_by_id(event_id)
        except AttributeError:
            event = self._get_event_by_id(event_id)

        self._current_event = event
        return deepcopy(event)

    def _get_event_by_index(self, index):
        """
        Method for extracting a particular event by looping through events
        until it finds the requested event index.
        If a file format allows random event access, then is can define its
        own `get_event_by_index` method in its
        `ctapipe.io.eventsource.EventSource` to allow this class to
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
        if self._getevent_warn:
            msg = (
                "Seeking event by iterating through events.. (potentially long process)"
            )
            self.log.warning(msg)
            self._getevent_warn = False

        if self._current_event and index < self._current_event.count:
            self._reset()

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
        `ctapipe.io.eventsource.EventSource` to allow this class to
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
        if self._getevent_warn:
            msg = (
                "Seeking event by iterating through events.. (potentially long process)"
            )
            self.log.warning(msg)
            self._getevent_warn = False

        self._reset()  # Event ids may not be in order, so always reset

        for event in self._source:
            if event.index.event_id == event_id:
                return event
        raise IndexError(f"Event id {event_id} not found in file")

    def __len__(self):
        """
        Method for getting number of events in file. By default this is
        obtained by looping through the file and counting the events. If a
        file format has a more efficient method of supplying this information,
        the `ctapipe.io.eventsource.EventSource` for that file format
        can define its own `__len__` method, which this class will then
        use instead.

        Returns
        -------
        self._n_events : int
            Number of events in the file
        """
        # Only need to calculate once
        if not self._n_events:
            try:
                count = len(self._event_source)
            except TypeError:
                self.log.warning(
                    "Obtaining length of file by looping through "
                    "all events... (potentially long process)"
                )
                count = 0
                for _ in self:
                    count += 1
            self._n_events = count
        return self._n_events
