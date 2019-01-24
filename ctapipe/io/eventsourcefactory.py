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
    The `EventSource` `ctapipe.core.factory.Factory`. This
    `ctapipe.core.factory.Factory` allows the correct
    `EventSource` to be obtained for the event file being read.

    This factory tests each EventSource by calling
    `EventSource.check_file_compatibility` to see which `EventSource`
    is compatible with the file.

    Using `EventFileReaderFactory` in a script allows it to be compatible with
    any file format that has an `EventSource` defined.

    To use within a `ctapipe.core.tool.Tool`:

    >>> from ctapipe.core.tool import Tool
    >>> class ExampleTool(Tool):
    >>>     def setup(self):
    >>>         kwargs = dict(config=self.config, tool=self)
    >>>         source = EventSourceFactory(**kwargs).get_product()
    >>>         print(source.__class__.__name__)
    >>>     def start(self, **kwargs):
    >>>         pass
    >>>     def finish(self, **kwargs):
    >>>         pass

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
    product : traitlets.CaselessStrEnum
        A string with the `EventSource.__name__` of the reader you want to
        use. If left blank, `EventSource.check_file_compatibility` will be
        used to find a compatible event_source.
    """
    base = EventSource
    product_help = ('EventSource to use. If None then a reader will '
                    'be chosen based on the input_url')

    input_url = Unicode(
        '',
        help='Path to the input file containing events.'
    ).tag(config=True)

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
        if self.input_url:
            constructor = self._get_constructor()
            kwargs = self._clean_kwargs_for_product(constructor, kwargs)
            instance = constructor(self.config, self.parent,
                                   input_url=self.input_url, **kwargs)
            return instance
        else:
            return super().get_product(**kwargs)


def event_source(input_url, config=None, parent=None, **kwargs):
    """
    Helper function to quickly construct an `EventSourceFactory` and produce
    an `EventSource`. This may be used in small scripts and demos for
    simplicity. In a `ctapipe.core.Tool` class, a `EventSourceFactory` should
    be manually constructed, so that the configuration info is correctly
    passed in.

    Examples
    --------
    >>> with event_source(url) as source:
    >>>    for event in source:
    >>>         print(event.r0.event_id)

    Parameters
    ----------
    input_url: str
        filename or URL pointing to an event file.

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
