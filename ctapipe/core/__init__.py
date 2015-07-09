from decorator import decorator

__all__ = ['component', 'Container']


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
    
    >>> data = Container("data")
    >>> data.add_item("x")
    >>> data.x = 3
    >>> print(data.x)
    3
    >>> print(data['x'])
    3

    """

    def __init__(self,name,**kwargs):
        """
        Parameters
        ----------
        self: type
            description
        name: str
            name of container instance
        kwargs: key=value
            initial data (`add_item` is called automatically for each)
        """
        self.add_item("_name", name)
        for key,val in kwargs.items():
            self.__dict__[key] = val

    @property
    def meta(self):
        """metadata associated with this container"""
        if not "_meta" in self.__dict__:
            self.add_item("_meta", Container("meta"))
        return self._meta
            
    def add_item(self, name,value=None):
        """
        Add a new item of data to this Container, initialized to None by
        default, or value if specified.
        """

        if name in self.__dict__:
            raise AttributeError("item '{}' is already in Container"
                                 .format(name))
        self.__dict__[name] = value
    
    def __setattr__(self, name, value):
        # prevent setting od values that are not yet registered
        if name not in self.__dict__:
            raise AttributeError("item '{}' doesn't exist in {}"
                                 .format(name, repr(self)))
        self.__dict__[name] = value
    
    def __getitem__(self, name):
        # allow getting value by string e.g. cont['x']
        return self.__dict__[name]
    
    def __str__(self, ):
        # string represnetation (e.g. `print(cont)`)
        return str(self.__dict__)

    def __repr__(self):
        # standard representation
        return '{0}.{1}("{2}", {3})'.format(self.__class__.__module__,
                                            self.__class__.__name__,
                                            self._name,
                                            ', '.join(self))

    def __iter__(self):
        # allow iterating over item names
        return (k for k in self.__dict__.keys() if not k.startswith("_"))
