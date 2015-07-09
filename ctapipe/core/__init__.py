from decorator import decorator

__all__ = ['component', 'Container']

import logging
logger = logging.getLogger(__name__)


#@decorator
def component():
    """ Component dectorator """
    pass


class Container:
    """Generic class that can hold and accumulate data to be passed
    between Components.

    Container members can be accessed like a dict or with . syntax.
    You can also iterate over the member names (useful for
    serialization). However, new data cannot be added arbitrarily. One
    must call `~ctapipe.core.Container.add_item` to add a new variable
    to the Container, otherwise an `AttributeError` will be thrown.
    
    >>> data = Container()
    >>> data.add_item("x")
    >>> data.x = 3
    >>> print(data.x)
    3
    >>> print(data['x'])
    3

    """

    def __init__(self,name,**kwargs):
        self.add_item("_name")
        self._name = name
        for key,val in kwargs.items():
            self.add_item(key)
            self[key] = val
    
    def add_item(self, name):
        """
        Add a new item of data to this Container, initialized to None by
        default.
        """

        if name in self.__dict__:
            raise AttributeError("item '{}' is already in Container"
                                 .format(name))
        logger.debug("added {}".format(name))
        self.__dict__[name] = None
    
    def __setattr__(self, name, value):
        if name not in self.__dict__:
            raise AttributeError("item '{}' doesn't exist in {}"
                                 .format(name, repr(self)))
        self.__dict__[name] = value
    
    def __getitem__(self, name):
        return self.__dict__[name]
    
    def __str__(self, ):
        return str(self.__dict__)

    def __repr__(self):
        return '{0}.{1}({2})'.format(self.__class__.__module__,
                                     self.__class__.__name__,
                                     ', '.join(self))

    def __iter__(self):
        return (k for k in self.__dict__.keys() if not k.startswith("_"))
