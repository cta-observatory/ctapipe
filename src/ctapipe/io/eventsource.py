"""
Handles reading of different event/waveform containing files
"""
import warnings
from abc import abstractmethod
from collections.abc import Generator

from traitlets.config.loader import LazyConfigValue

from ctapipe.atmosphere import AtmosphereDensityProfile

from ..containers import (
    ArrayEventContainer,
    ObservationBlockContainer,
    SchedulingBlockContainer,
    SimulatedShowerDistribution,
    SimulationConfigContainer,
)
from ..core import ToolConfigurationError
from ..core.component import Component, find_config_in_hierarchy
from ..core.traits import CInt, Int, Path, Set, TraitError, Undefined
from ..instrument import SubarrayDescription
from .datalevels import DataLevel

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
    regardless of the file format or data origin.

    EventSource itself is an abstract class, but will create an
    appropriate subclass if a compatible source is found for the given
    ``input_url``.

    >>> EventSource(input_url="dataset://gamma_prod5.simtel.zst")
    <ctapipe.io.simteleventsource.SimTelEventSource ...>

    An ``EventSource`` can also be created through the configuration system,
    by passing ``config`` or ``parent`` as appropriate.
    E.g. if using ``EventSource`` inside of a ``Tool``, you would do:

    >>> self.source = EventSource(parent=self) # doctest: +SKIP

    To loop through the events in a file:

    >>> source = EventSource(input_url="dataset://gamma_prod5.simtel.zst", max_events=2)
    >>> for event in source:
    ...     print(event.count)
    0
    1

    **NOTE**: Every time a new loop is started through the source,
    it tries to restart from the first event, which might not be supported
    by the event source.

    It is encouraged to use ``EventSource`` in a context manager to ensure
    the correct cleanups are performed when you are finished with the source:

    >>> with EventSource(input_url="dataset://gamma_prod5.simtel.zst", max_events=2) as source:
    ...    for event in source:
    ...        print(event.count)
    0
    1

    **NOTE**: EventSource implementations should not reuse the same ArrayEventContainer,
    as these are mutable and may lead to errors when analyzing multiple events.


    Parameters
    ----------
    input_url : str | Path
        Path to the input event file.
    max_events : int
        Maximum number of events to loop through in generator
    allowed_tels: Set or None
        Ids of the telescopes to be included in the data.
        If given, only this subset of telescopes will be present in the
        generated events. If None, all available telescopes are used.
    """

    #: ctapipe_io entry points may provide EventSource implementations
    plugin_entry_point = "ctapipe_io"

    input_url = Path(help="Path to the input file containing events.").tag(config=True)

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
    def simulation_config(self) -> dict[int, SimulationConfigContainer] | None:
        """The simulation configurations of all observations provided by the
        EventSource, or None if the source does not provide simulated data

        Returns
        -------
        dict[int,ctapipe.containers.SimulationConfigContainer] | None
        """
        return None

    @property
    @abstractmethod
    def observation_blocks(self) -> dict[int, ObservationBlockContainer]:
        """
        Obtain the ObservationConfigurations from the EventSource, indexed by obs_id
        """
        pass

    @property
    @abstractmethod
    def scheduling_blocks(self) -> dict[int, SchedulingBlockContainer]:
        """
        Obtain the ObservationConfigurations from the EventSource, indexed by obs_id
        """
        pass

    @property
    @abstractmethod
    def is_simulation(self) -> bool:
        """
        Whether the currently opened file is simulated

        Returns
        -------
        bool

        """

    @property
    @abstractmethod
    def datalevels(self) -> tuple[DataLevel]:
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
    def obs_ids(self) -> list[int]:
        """
        The observation ids of the runs located in the file
        Unmerged files should only contain a single obs id.

        Returns
        -------
        list[int]
        """
        return list(self.observation_blocks.keys())

    @property
    def atmosphere_density_profile(self) -> AtmosphereDensityProfile | None:
        """atmosphere density profile that can be integrated to
        convert between h_max and X_max.  This should correspond
        either to what was used in a simulation, or a measurement
        for use with observed data.

        Returns
        -------
        AtmosphereDensityProfile:
           profile to be used
        """
        return None

    @property
    def simulated_shower_distributions(self) -> dict[int, SimulatedShowerDistribution]:
        """
        The distribution of simulated showers for each obs_id.

        Returns
        -------
        dict[int,ctapipe.containers.SimulatedShowerDistribution]
        """
        return {}

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

        # validate input url with the traitlet validate method
        # to make sure it's compatible and to raise the correct error
        input_url = EventSource.input_url.validate(obj=None, value=input_url)

        available_classes = cls.non_abstract_subclasses()

        missing_deps = {}
        for name, subcls in available_classes.items():
            try:
                if subcls.is_compatible(input_url):
                    return subcls
            except ModuleNotFoundError as e:
                missing_deps[subcls] = e.module
            except Exception as e:
                warnings.warn(f"{name}.is_compatible raised exception: {e}")

        # provide a more helpful error for non-existing input_url
        if not input_url.exists():
            raise TraitError(
                f"input_url {input_url} is not an existing file "
                " and no EventSource implementation claimed compatibility."
            )

        available_sources = [
            name for name, cls in available_classes.items() if cls not in missing_deps
        ]

        msg = (
            f"Could not find compatible EventSource for input_url: {input_url!r}\n"
            f"in available EventSources: {available_sources}\n"
            "EventSources that are installed but could not be used due to missing dependencies:\n\t"
            + "\n\t".join(
                f"{source.__name__}: {missing}"
                for source, missing in missing_deps.items()
            )
        )
        raise ValueError(msg)

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
            # first look at appropriate position in the config hierarchy
            input_url = find_config_in_hierarchy(parent, "EventSource", "input_url")

            # if not found, check top level
            if isinstance(input_url, LazyConfigValue):
                if not isinstance(parent.config.EventSource.input_url, LazyConfigValue):
                    input_url = parent.config.EventSource.input_url
                else:
                    input_url = cls.input_url.default_value

        return input_url

    def close(self):
        """Close this event source.

        No-op by default. Should be overridden by sources needing a cleanup-step
        """
        pass
