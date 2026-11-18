"""
Handles seeking to a particular event in a `ctapipe.io.EventSource`
"""

from copy import deepcopy

from ctapipe.core import Component

__all__ = ["EventSeeker"]


class EventSeeker(Component):
    """
    Find a particular event in an event source.

    Events can be requested by sequential index or event ID. A source-provided lookup is
    used when available. Otherwise the seeker iterates over the events, which can be
    slow for large files.

    Pass a `~ctapipe.io.EventSource` instance, such as
    `~ctapipe.io.SimTelEventSource`. For sequential sources, use a back-seekable
    input when a lookup may need to restart from the beginning.

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

    The event index is the sequential ``event.count`` assigned by ctapipe.
    The event ID is the identifier supplied by the input file.
    """

    def __init__(self, event_source, config=None, parent=None, **kwargs):
        """
        Initialize the event seeker with an input source.

        Retain the event source and initialize an iterator for sequential lookups.

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
        self._source = iter(self._event_source)
        self._current_event = None
        self._has_fast_seek = False  # By default seeking iterates through
        self._getevent_warn = True

    def _reset(self):
        """
        Recreate the generator so it starts from the beginning
        """
        if self._event_source.is_stream:
            raise OSError("Back-seeking is not possible for event source")
        self._source = iter(self._event_source)
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
        Scan the event iterator for the requested index.

        Reset the iterator when seeking behind the current event. Use `get_event_index`
        as the public entry point, which can use source-provided random access.

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
        Scan the event iterator for the requested ID.

        Reset first if an event was already read, since IDs need not be in order. Use
        `get_event_id` as the public entry point, which can use source-provided random
        access.

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

        if self._current_event is not None:
            self._reset()  # Event ids may not be in order, so always reset

        for event in self._source:
            self._current_event = event
            if event.index.event_id == event_id:
                return event
        raise IndexError(f"Event id {event_id} not found in file")

    def __len__(self):
        """
        Count the events accessible through the event source.

        Use the source's length when available. Otherwise count events by iteration,
        which requires a non-streaming source. A nonzero count is cached for later
        calls.

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
