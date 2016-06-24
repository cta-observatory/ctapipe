from traitlets.config import Configurable
from logging import getLogger


class Component(Configurable):
    """This should be the base class of all Components (sometimes called
    workers, makers, etc). Before adding any class methods, you should
    define the `name` and the user-configurable *parameters*.

    Components use `traitlets` for all configurable parameters, which
    can include validators, default values, and descriptions. These
    will be automatically translated into configuration parameters
    (command-line, config file, etc). Any parameter that should be
    externally configurable must have it's `config` attribute set to `True`,
    e.g. defined like `myparam = Integer(default=0, help='the
    parameter').tag(config=True)`.

    All components also contain a `logger` instance, that you must use
    to output info, debugging data, warnings, etc (do not use
    `print()` statements).



    ```
    from ctapipe.core import Component
    from traitlets import Integer, Float

    class MyComponent(Component):
        name = "MyComponent"
        value = Integer(default

    ```

    """

    def __init__(self, parent, **kwargs):
        """
        Parameters
        ----------
        self: type
            description
        parent: Tool or Component
            Tool or component that is the Parent of this one
        kwargs: type
            other paremeters

        """

        super().__init__(parent=parent, **kwargs)

        # set up logging
        if self.parent:
            self.log = self.parent.log.getChild(self.__class__.__name__)
        else:
            self.log = getLogger(self.__class__.__name__)

