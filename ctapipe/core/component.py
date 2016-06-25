from traitlets.config import Configurable
from logging import getLogger


class Component(Configurable):
    """This should be the base class of all Components (sometimes called
    workers, makers, etc). Before adding any class methods, you should
    define any user-configurable *parameters* as `traitlets`.

    `traitlets` can validate values and provide defaults and
    descriptions. These will be automatically translated into
    configuration parameters (command-line, config file, etc). Note
    that any parameter that should be externally configurable must
    have its `config` attribute set to `True`, e.g. defined like
    `myparam = Integer(0, help='the parameter').tag(config=True)`.

    All components also contain a `logger` instance in their `log`
    attribute, that you must use to output info, debugging data,
    warnings, etc (do not use `print()` statements, instead use
    `self.log.info()`, `self.log.warn()`, `self.log.debug()`, etc).


    ```
    from ctapipe.core import Component
    from traitlets import (Integer, Float)

    class MyComponent(Component):
        description = "Does something"
        some_option = Integer(default_value=6,
                              help='a value to set').tag(config=True)


    comp = MyComponent(None)
    comp.some_option = 6      # ok
    comp.some_option = 'test' # will fail validation

    ```

    """

    def __init__(self, parent, **kwargs):
        """
        Parameters
        ----------
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
