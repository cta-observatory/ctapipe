from ctapipe.core.factory import Factory, child_subclasses
from ctapipe.io.eventsource import EventSource
from traitlets import Unicode

# EventFileReader imports so that EventFileReaderFactory can see them
# (they need to exist in the global namespace)
import ctapipe.io.hessioeventsource
from . import sst1meventsource
from . import nectarcameventsource
from . import lsteventsource
import ctapipe.io.targetioeventsource


__all__ = ['EventSourceFactory', 'event_source']


class EventSourceFactory(Factory):
    """
    Allows the correct`EventSource` to be obtained for the event file
    being read.

    This factory finds the first compatible EventSource by looping over all
    EventSources in the global namespace and calling
    `EventSource.is_compatible`. This compatible EventSource is then returned,
    with the file already opened by it.

    Using `EventSourceFactory` in a script allows it to be compatible with
    any file format that has an `EventSource` defined.

    An example of a simple use of this class:

    >>> from ctapipe.utils import get_dataset_path
    >>> url = get_dataset_path("gamma_test.simtel.gz")
    >>> source = EventSourceFactory(input_url=url).get_product()

    An example of use within a `ctapipe.core.tool.Tool`:

    >>> from ctapipe.core.tool import Tool
    >>> from traitlets import Dict, List
    >>>
    >>> class ExampleTool(Tool):
    >>>
    >>>     aliases = Dict(dict(
    >>>        f='EventSourceFactory.input_url',
    >>>     ))
    >>>     classes = List([
    >>>        EventSourceFactory,
    >>>     ])
    >>>
    >>>     def setup(self):
    >>>         kwargs = dict(config=self.config, tool=self)
    >>>         source = EventSourceFactory(**kwargs).get_product()
    >>>         print(source.__class__.__name__)
    >>>
    >>>     def start(self, **kwargs):
    >>>         pass
    >>>
    >>>     def finish(self, **kwargs):
    >>>         pass
    >>>
    >>> if __name__ == '__main__':
    >>>     exe = ExampleTool()
    >>>     exe.run()

    Running the above as a script from the commandline with the argument
    "-f path/to/file.simtel.gz" will pass the input_url to the
    EventSourceFactory.

    It is possible to force the returned EventSource via the product
    argument/traitlet:

    >>> from ctapipe.utils import get_dataset_path
    >>> path = get_dataset_path("gamma_test.simtel.gz")
    >>> source = EventSourceFactory(
    >>>    input_url=path,
    >>>    product="HESSIOEventSource"
    >>> ).get_product()
    """
    base = EventSource
    product_help = ('EventSource to use. If None then a reader will '
                    'be chosen based on the input_url')

    input_url = Unicode(
        '',
        help='Path to the input file containing events.'
    ).tag(config=True)

    def __init__(self, input_url=None, max_events=None, allowed_tels=None,
                 config=None, tool=None, **kwargs):
        """
        Parameters
        ----------
        input_url : str
            URL to a file to read.
            The input_url can be specified either via an argument to
            __init__, or via the command line utilising the input_url traitlet
            from a `ctapipe.core.tool.Tool` (see Tool example above).
            The traitlet is then passed to this Factory via the config
            argument.
            If specified as a named argument, the input_url traitlet will
            overriden.
        max_events : int
            Maximum number of events to read from file
        allowed_tels : set
            Set of allowed tel_ids, others will be ignored. If left empty, all
            telescopes in the input stream will be included
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            This argument is typically only used from within a
            `ctapipe.core.Tool`.
            Used to set traitlet values.
            Leave as None if no configuration to pass.
        tool : ctapipe.core.Tool or None
            Tool executable that is calling this component.
            This argument is typically only used from within a
            `ctapipe.core.Tool`.
            Passes the correct logger to the component.
            Leave as None if no Tool to pass.
        kwargs
            Named arguments for the EventSourceFactory. These are not passed
            on to the EventSource.
        """
        if input_url:
            kwargs['input_url'] = input_url
        self.max_events = max_events
        self.allowed_tels = set() if allowed_tels is None else allowed_tels
        super().__init__(config=config, tool=tool, **kwargs)

    def _get_product_name(self):
        """
        Method to obtain the correct name for the EventSource. If the
        `product` traitlet is set, then the correspondingly named EventSource
        is returned. If the `product` traitlet is unset, the subclasses to
        EventSource are looped through, executing `is_compatible` with the
        `input_url` traitlet until a compatible EventSource is found.

        Returns
        -------
        str
            The name of the EventSource to return from the EventSourceFactory.
        """
        try:
            # If a specific EventSource is requested via the `product` traitlet
            return super()._get_product_name()
        except AttributeError:
            if not self.input_url:
                raise ValueError("Please specify an input_url for event file")

            # Find compatible EventSource
            subclasses = child_subclasses(self.base)
            for subclass in subclasses.values():
                if subclass.is_compatible(self.input_url):
                    return subclass.__name__
            raise ValueError(
                "Cannot find compatible EventSource for \n"
                "\turl: {}\n"
                "in available EventSources:\n"
                "\t{}".format(self.input_url, list(subclasses.keys()))
            )


    def get_product(self):
        """
        Obtain the correct EventSource for the input_url supplied to this
        EventSourceFactory (via the arguments to __init__ or via the
        Tool config).

        Parameters
        ----------
        kwargs
            Named arguments to pass to the EventSource. The path to the file
            is passed to the EventSource automatically, and should not be
            specified here.

        Returns
        -------
        product_instance : ctapipe.io.EventSource
            EventSource corresponding to the input_url, or corresponding to
            the product argument/traitlet if specified.
        """
        product_name = self._get_product_name()
        product_constructor = self._get_product_constructor(product_name)
        kwargs = dict(input_url=self.input_url)
        if self.max_events:
            kwargs['max_events'] = self.max_events
        if self.allowed_tels:
            kwargs['allowed_tels'] = self.allowed_tels
        product_instance = product_constructor(
            self.config, self.parent, **kwargs
        )
        return product_instance


    @classmethod
    def produce(cls, config=None, tool=None, **kwargs):
        """
        Deprecated method for obtaining an EventSource from an
        EventSourceFactory via the classmethod.

        Instead, one should switch to the new API using the `get_product()`
        method:

        >>> from ctapipe.utils import get_dataset_path
        >>> url = get_dataset_path("gamma_test.simtel.gz")
        >>> source = EventSourceFactory(input_url=url).get_product()

        See the EventSourceFactory class docstring for more examples.

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            This argument is typically only used from within a
            `ctapipe.core.Tool`.
            Used to set traitlet values.
            Leave as None if no configuration to pass.
        tool : ctapipe.core.Tool or None
            Tool executable that is calling this component.
            This argument is typically only used from within a
            `ctapipe.core.Tool`.
            Passes the correct logger to the component.
            Leave as None if no Tool to pass.
        kwargs

        Returns
        -------

        """
        return super().produce(config=config, tool=tool, **kwargs)


def event_source(input_url, max_events=None, allowed_tels=None,
                 config=None, parent=None, **kwargs):
    """
    Helper function to quickly construct an `EventSourceFactory` and produce
    an `EventSource`. This may be used in small scripts and demos for
    simplicity. In a `ctapipe.core.Tool` class, a `EventSourceFactory` should
    be manually constructed, so that the configuration info is correctly
    passed in.

    Examples
    --------
    >>> from ctapipe.utils import get_dataset_path
    >>> url = get_dataset_path("gamma_test.simtel.gz")
    >>> with event_source(url) as source:
    >>>    for event in source:
    >>>         print(event.r0.event_id)

    Parameters
    ----------
    input_url: str
        Filename or URL pointing to an event file
    max_events : int
        Maximum number of events to read from file
    allowed_tels : set
        Set of allowed tel_ids, others will be ignored. If left empty, all
        telescopes in the input stream will be included
    config : traitlets.loader.Config
        Configuration specified by config file or cmdline arguments.
        This argument is typically only used from within a `ctapipe.core.Tool`.
        Used to set traitlet values.
        Leave as None if no configuration to pass.
    tool : ctapipe.core.Tool or None
        Tool executable that is calling this component.
        This argument is typically only used from within a `ctapipe.core.Tool`.
        Passes the correct logger to the component.
        Leave as None if no Tool to pass.
    kwargs
        Named arguments for the EventSourceFactory. These are not passed
        on to the EventSource.

    Returns
    -------
    EventSource:
        a properly constructed `EventSource` subclass, depending on the
        input filename.
    """

    reader = EventSourceFactory(
        input_url=input_url, max_events=max_events, allowed_tels=allowed_tels,
        config=config, tool=parent, **kwargs
    ).get_product()

    return reader
