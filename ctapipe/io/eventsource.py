"""
Handles reading of different event/waveform containing files
"""
from abc import abstractmethod
from traitlets.config.loader import LazyConfigValue
from typing import Tuple, List, Generator

from ..instrument import SubarrayDescription
from .datalevels import DataLevel
from ..containers import ArrayEventContainer
from ..core import ToolConfigurationError, Provenance
from ..core.component import Component, non_abstract_children, find_config_in_hierarchy
from ..core.traits import Path, Int, CInt, Set, Undefined


__all__ = ["EventSource"]


class EventSource(Component):
    """
    Parent class for EventSources.

    EventSources read input files and generate `~ctapipe.containers.ArrayEventContainer`
    instances when iterated over.

    A new EventSource should be created for each type of event file read
    into ctapipe, e.g. sim_telarray files are read by the `~ctapipe.io.SimTelEventSource`.

    EventSource provides a common high-level interface for accessing event
    information from different data sources (simulation or different camera
    file formats). Creating an EventSource for a new
    file format or other event source ensures that data can be accessed in a common way,
    irregardless of the file format or data origin.

    EventSource itself is an abstract class, but will create an
    appropriate subclass if a compatible source is found for the given
    ``input_url``.

    >>> EventSource(input_url="dataset://gamma_test_large.simtel.gz")
    <ctapipe.io.simteleventsource.SimTelEventSource ...>

    An ``EventSource`` can also be created through the configuration system,
    by passing ``config`` or ``parent`` as appropriate.
    E.g. if using ``EventSource`` inside of a ``Tool``, you would do:
    >>> self.source = EventSource(parent=self) # doctest: +SKIP

    To loop through the events in a file:
    >>> source = EventSource(input_url="dataset://gamma_test_large.simtel.gz", max_events=2)
    >>> for event in source:
    ...     print(event.count)
    0
    1

    **NOTE**: Every time a new loop is started through the source,
    it tries to restart from the first event, which might not be supported
    by the event source.

    It is encouraged to use ``EventSource`` in a context manager to ensure
    the correct cleanups are performed when you are finished with the source:

    >>> with EventSource(input_url="dataset://gamma_test_large.simtel.gz", max_events=2) as source:
    ...    for event in source:
    ...        print(event.count)
    0
    1

    **NOTE**: For effiency reasons, most sources only use a single ``ArrayEvent`` instance
    and update it with new data on iteration, which might lead to surprising
    behaviour if you want to access multiple events at the same time.
    To keep an event and prevent its data from being overwritten with the next event's data,
    perform a deepcopy: ``some_special_event = copy.deepcopy(event)``.


    Attributes
    ----------
    input_url : str
        Path to the input event file.
    max_events : int
        Maximum number of events to loop through in generator
    allowed_tels: Set or None
        Ids of the telescopes to be included in the data.
        If given, only this subset of telescopes will be present in the
        generated events. If None, all available telescopes are used.
    """

    input_url = Path(
        directory_ok=False,
        exists=True,
        help="Path to the input file containing events.",
    ).tag(config=True)

    max_events = Int(
        None,
        allow_none=True,
        help="Maximum number of events that will be read from the file",
    ).tag(config=True)

    allowed_tels = Set(
        trait=CInt(),
        default_value=None,
        allow_none=True,
        help=(
            "list of allowed tel_ids, others will be ignored. "
            "If None, all telescopes in the input stream "
            "will be included"
        ),
    ).tag(config=True)

    def __new__(cls, input_url=Undefined, config=None, parent=None, **kwargs):
        """
        Returns a compatible subclass for given input url, either
        directly or via config / parent
        """
        # needed to break recursion, as __new__ of subclass will also
        # call this method
        if cls is not EventSource:
            return super().__new__(cls)

        # check we have at least one of these to be able to determine the subclass
        if input_url in {None, Undefined} and config is None and parent is None:
            raise ValueError("One of `input_url`, `config`, `parent` is required")

        if input_url in {None, Undefined}:
            input_url = cls._find_input_url_in_config(config=config, parent=parent)

        subcls = cls._find_compatible_source(input_url)
        return super().__new__(subcls)

    def __init__(self, input_url=None, config=None, parent=None, **kwargs):
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
        # traitlets differentiates between not getting the kwarg
        # and getting the kwarg with a None value.
        # the latter overrides the value in the config with None, the former
        # enables getting it from the config.
        if input_url not in {None, Undefined}:
            kwargs["input_url"] = input_url

        super().__init__(config=config, parent=parent, **kwargs)

        self.metadata = dict(is_simulation=False)
        self.log.info(f"INPUT PATH = {self.input_url}")

        if self.max_events:
            self.log.info(f"Max events being read = {self.max_events}")

        Provenance().add_input_file(str(self.input_url), role="DL0/Event")

    @staticmethod
    @abstractmethod
    def is_compatible(file_path):
        """
        Abstract method to be defined in child class.

        Perform a set of checks to see if the input file is compatible
        with this file event_source.

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

    @property
    @abstractmethod
    def subarray(self) -> SubarrayDescription:
        """
        Obtain the subarray from the EventSource

        Returns
        -------
        ctapipe.instrument.SubarrayDecription

        """

    @property
    @abstractmethod
    def is_simulation(self) -> bool:
        """
        Weither the currently opened file is simulated

        Returns
        -------
        bool

        """

    @property
    @abstractmethod
    def datalevels(self) -> Tuple[DataLevel]:
        """
        The datalevels provided by this event source

        Returns
        -------
        tuple[ctapipe.io.DataLevel]
        """

    def has_any_datalevel(self, datalevels) -> bool:
        """
        Check if any of `datalevels` is in self.datalevels

        Parameters
        ----------
        datalevels: Iterable
            Iterable of datalevels
        """
        return any(dl in self.datalevels for dl in datalevels)

    @property
    @abstractmethod
    def obs_ids(self) -> List[int]:
        """
        The observation ids of the runs located in the file
        Unmerged files should only contain a single obs id.

        Returns
        -------
        list[int]
        """

    @abstractmethod
    def _generator(self) -> Generator[ArrayEventContainer, None, None]:
        """
        Abstract method to be defined in child class.

        Generator where the filling of the `ctapipe.containers` occurs.

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
            yield event
            if self.max_events and event.count >= self.max_events - 1:
                break

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @classmethod
    def _find_compatible_source(cls, input_url):
        if input_url == "" or input_url in {None, Undefined}:
            raise ToolConfigurationError("EventSource: No input_url was specified")

        # validate input url with the traitel validate method
        # to make sure it's compatible and to raise the correct error
        input_url = EventSource.input_url.validate(obj=None, value=input_url)

        available_classes = non_abstract_children(cls)

        for subcls in available_classes:
            if subcls.is_compatible(input_url):
                return subcls

        raise ValueError(
            "Cannot find compatible EventSource for \n"
            "\turl:{}\n"
            "in available EventSources:\n"
            "\t{}".format(input_url, [c.__name__ for c in available_classes])
        )

    @classmethod
    def from_url(cls, input_url, **kwargs):
        """
        Find compatible EventSource for input_url via the `is_compatible`
        method of the EventSource

        Parameters
        ----------
        input_url : str
            Filename or URL pointing to an event file
        kwargs
            Named arguments for the EventSource

        Returns
        -------
        instance
            Instance of a compatible EventSource subclass
        """
        subcls = cls._find_compatible_source(input_url)
        return subcls(input_url=input_url, **kwargs)

    @classmethod
    def _find_input_url_in_config(cls, config=None, parent=None):
        if config is None and parent is None:
            raise ValueError("One of config or parent must be provided")

        if config is not None and parent is not None:
            raise ValueError("Only one of config or parent must be provided")

        input_url = None

        # config was passed
        if config is not None:
            if not isinstance(config.input_url, LazyConfigValue):
                input_url = config.input_url
            elif not isinstance(config.EventSource.input_url, LazyConfigValue):
                input_url = config.EventSource.input_url
            else:
                input_url = cls.input_url.default_value

        # parent was passed
        else:
            # first look at appropriate position in the config hierarcy
            input_url = find_config_in_hierarchy(parent, "EventSource", "input_url")

            # if not found, check top level
            if isinstance(input_url, LazyConfigValue):
                if not isinstance(parent.config.EventSource.input_url, LazyConfigValue):
                    input_url = parent.config.EventSource.input_url
                else:
                    input_url = cls.input_url.default_value

        return input_url

    @classmethod
    def from_config(cls, config=None, parent=None, **kwargs):
        """
        Find compatible EventSource for the EventSource.input_url traitlet
        specified via the config.

        This method is typically used in Tools, where the input_url is chosen via
        the command line using the traitlet configuration system.

        Parameters
        ----------
        config : traitlets.config.loader.Config
            Configuration created in the Tool
        kwargs
            Named arguments for the EventSource

        Returns
        -------
        instance
            Instance of a compatible EventSource subclass
        """
        input_url = cls._find_input_url_in_config(config=config, parent=parent)
        return cls.from_url(input_url, config=config, parent=parent, **kwargs)

    def close(self):
        """Close this event source.

        No-op by default. Should be overriden by sources needing a cleanup-step
        """
        pass
