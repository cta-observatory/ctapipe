# Licensed under a 3-clause BSD style license - see LICENSE.rst

from pprint import pformat
from astropy.table import Table
from astropy.units import Quantity
from ctapipe.io.files import get_file_type
import numpy as np
import logging

__all__ = [
    'component',
    'Container',
]


#@decorator
def component():
    """Component decorator"""
    pass


class Container:

    """Generic class that can hold and accumulate data to be passed
    between Components.

    The purpose of this class is to provide a flexible data structure
    that works a bit like a dict or blank Python class, but prevents
    the user from accessing members that have not been defined
    a-priori (more like a C struct).  Generally, one can make a
    sub-class and "Register" all of the members that should be there
    in the `__init__` method by calling `~Container.add_item`.

    Container members can be accessed like a dict `container['item']
    or with `continer.item` syntax.  You can also iterate over the
    member names (useful for serialization). However, new data cannot
    be added arbitrarily. One must call
    `~ctapipe.core.Container.add_item` to add a new variable to the
    Container, otherwise an `AttributeError` will be thrown.

    Parameters
    ----------
    self: type
        description
    name: str
        name of container instance
    kwargs: key=value
        initial data (`add_item` is called automatically for each)

    Example
    -------
    >>> data = Container("data")
    >>> data.add_item("x")
    >>> data.x = 3
    >>> print(data.x)
    3
    >>> print(data['x'])
    3

    """

    def __init__(self, name="Container", **kwargs):
        self.add_item("_name", name)
        for key, val in kwargs.items():
            self.__dict__[key] = val

    @property
    def meta(self):
        """metadata associated with this container"""
        if "_meta" not in self.__dict__:
            self.add_item("_meta", Container("meta"))
        return self._meta

    def add_item(self, name, value=None):
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
        # string representation (e.g. `print(cont)`)
        return pformat(self.__dict__)

    def __repr__(self):
        # standard representation
        return '{0}.{1}("{2}", {3})'.format(self.__class__.__module__,
                                            self.__class__.__name__,
                                            self._name,
                                            ', '.join(self))

    def __iter__(self):
        # allow iterating over item names
        return (k for k in self.__dict__.keys() if not k.startswith("_"))

    def as_dict(self):
        '''Creates a dictionary of Container items unrolling recursively
        nested containers.'''
        d = dict()
        for k, v in self.items():
            if isinstance(v, Container):
                d[k] = v.as_dict()
                continue
            d[k] = v
        return d

    def items(self):
        '''Iterate over pairs of key, value. Just like the dictionary method'''
        # allow iterating over item names
        return ((k, v) for k, v in self.__dict__.items()
                if not k.startswith('_'))

    def to_table(self):
        '''Create Table from Container'''
        # Scalar `Quantity` objects do not have __len__ method which is
        # needed by Table.write. We artificially change their shape
        # With chunking this should not be an issue
        for _, val in self.items():
            if isinstance(val, Quantity) and val.isscalar:
                val.shape = 1

        names = [i.upper() for i in self]
        dtype = [v.dtype for _, v in self.items()]
        data = [v for _, v in self.items()]
        # data = [v for _, v.chunk in self.items()] # It depends on
                                                    # chunking syntax

        return Table(data=data,
                     names=names,
                     dtype=dtype,
                     meta=self.meta.as_dict())

    def write(self, *args, **kwargs):
        '''Write table using astropy.table write method'''
        # if self._meta is None:
        #     logging.error("Metadata should be present before writing data")
        #     return

        table = self.to_table()
        # Write HDU name
        if get_file_type(args[0]) == "fits":
            table.meta["EXTNAME"] = self._name

        table.write(*args, **kwargs)

        return table