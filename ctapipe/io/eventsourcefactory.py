from ctapipe.core.factory import Factory
from ctapipe.io.eventsource import EventSource

# EventFileReader imports so that EventFileReaderFactory can see them
# (they need to exist in the global namespace)
import ctapipe.io.hessioeventsource


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

    >>> reader = EventSourceFactory.produce(config=self.config, tool=self)

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
        used to find a compatible reader.
    """
    base = EventSource
    custom_product_help = ('EventSource to use. If None then a reader will '
                           'be chosen based on the input_url')
    input_url = None  # Instanced as a traitlet by FactoryMeta

    def _get_product_name(self):
        try:
            return super()._get_product_name()
        except AttributeError:
            if self.input_url is None:
                raise ValueError("Please specify an input_url for event file")
            try:
                for subclass in self.subclasses:
                    print(subclass)
                    print(self.input_url)
                    print(subclass.is_compatible(self.input_url))
                    if subclass.is_compatible(self.input_url):
                        return subclass.__name__
                raise ValueError
            except ValueError:
                self.log.exception("Cannot find compatible EventSource "
                                   "for: {}".format(self.input_url))
                raise
