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

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Parameters
        ----------
        input_url : str
            Path to a file to read.
            The input_url can be specified either via a named argument to
            __init__, or via the command line utilising the input_url traitlet
            from a `ctapipe.core.tool.Tool` (see Tool example above).
            The traitlet is passed to this Factory via the config argument.
            If specified as a named argument, the input_url traitlet will
            overriden.
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
        super().__init__(config=config, tool=tool, **kwargs)

    def _get_product_name(self):
        try:
            return super()._get_product_name()
        except AttributeError:
            if not self.input_url:
                raise ValueError("Please specify an input_url for event file")
            try:
                for subclass in child_subclasses(self.base).values():
                    if subclass.is_compatible(self.input_url):
                        return subclass.__name__
                raise ValueError
            except ValueError:
                self.log.exception("Cannot find compatible EventSource "
                                   "for: {}".format(self.input_url))
                raise

    def get_product(self, **kwargs):
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
        instance : ctapipe.io.EventSource
            EventSource corresponding to the input_url, or corresponding to
            the product argument/traitlet if specified.
        """
        if self.input_url:
            constructor = self._get_constructor()
            kwargs = self._clean_kwargs_for_product(constructor, kwargs)
            instance = constructor(self.config, self.parent,
                                   input_url=self.input_url, **kwargs)
            return instance
        else:
            return super().get_product(**kwargs)

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

def event_source(input_url, config=None, parent=None, **kwargs):
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
        filename or URL pointing to an event file.
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

    Returns
    -------
    EventSource:
        a properly constructed `EventSource` subclass, depending on the
        input filename.
    """

    reader = EventSourceFactory(
        config, parent, input_url=input_url
    ).get_product(**kwargs)

    return reader
