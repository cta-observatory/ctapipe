from pprint import pformat
from copy import copy

class Container:
    """Generic class that can hold and accumulate data to be passed
    between Components.

    The purpose of this class is to provide a flexible data structure
    that works a bit like a dict or blank Python class, but prevents
    the user from accessing members that have not been defined
    a-priori (more like a C struct). It is also used to transform the
    data into something that can be written to an output table.

    To use this class, all members must be defined as `Item`s with
    default values specified.  For hierarchical data structures, Items
    can use `Container` subclasses or a `Map` as the default value.

    >>>    class MyContainer(Container):
    >>>        x = Item(100,"The X value")
    >>>        energy = Item(-1, "Energy measurement", unit=u.TeV)
    >>>
    >>>    cont = MyContainer()
    >>>    print(cont.x)
    100

    """

    def __init__(self):
        object.__setattr__(self, "_metadata", dict())
        self.reset()

    def __setattr__(self, name, value):
        """Prevent new attributes that aren't in the class definition"""
        if hasattr(self.__class__, name):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                "{} has no attribute '{}'".format(self.__class__, name))

    @property
    def meta(self):
        return self._metadata

    @property
    def attributes(self):
        """
        Returns a dictionary of the attribute metadata of each item in the
        container class as a dict of `Item`s
        """
        return {key: val for key, val in self.__class__.__dict__.items()
                if isinstance(val, Item)}

    def items(self):
        """dict-like access"""
        return self.__dict__.items()

    def as_dict(self, recursive=False, flatten=False):
        """
        convert the `Container` into a dictionary

        Parameters
        ----------
        recursive: bool
            sub-Containers should also be converted to dicts
        flatten: type
            return a flat dictionary, with any sub-item keys generated
            by appending the sub-Container name.
        """
        if not recursive:
            return dict(self.__dict__.items())
        else:
            d = dict()
            for key, val in self.items():
                if isinstance(val, Container) or isinstance(val, Map):
                    if flatten:
                        d.update({"{}_{}".format(key, k): v
                                  for k, v in val.as_dict(recursive).items()})
                    else:
                        d[key] = val.as_dict(recursive, flatten)
                    continue
                d[key] = val
            return d

    @classmethod
    def disable_attribute_check(cls):
        """
        Globally turn off attribute checking for all Containers,
        which provides a ~5-10x speed up for setting attributes.
        This may be used e.g. after code is tested to speed up operation.
        """
        cls.__setattr__ = object.__setattr__

    def reset(self, recursive=True):
        """ set all values back to their default values"""
        for name, value in self.__class__.__dict__.items():
            if isinstance(value, Item):
                self.__dict__[name] = copy(value.default)
            if recursive and isinstance(value, Container):
                value.reset()

    def __str__(self):
        return pformat(self.as_dict(recursive=True))


class Map(dict):
    """A dictionary of sub-containers that can be added to a
    Container. This may be used e.g. to store telescope-wise
    Containers(e.g. indexed by `tel_id` or something similar that can
    be added to a containre like a normal `Item`.
    """

    def as_dict(self, recursive=False, flatten=False):
        if not recursive:
            return dict(self.items())
        else:
            d = dict()
            for key, val in self.items():
                if isinstance(val, Container):
                    if flatten:
                        d.update({"{}_{}".format(key, k): v
                                  for k, v in val.as_dict(recursive).items()})
                    else:
                        d[key] = val.as_dict(recursive, flatten)
                    continue
                d[key] = val
            return d

    def reset(self, recursive=True):
        for key, val in self.items():
            if isinstance(val, Container):
                val.reset(recursive=True)


class Item:
    """
    Defines the metadata associated with a value in a Container

    Parameters
    ----------
    default:
        default value of the item (this will be set when the `Container`
        is constructed, as well as when  `Container.reset()` is called
    description: str
        Help text associated with the item
    unit: `astropy.units.Quantity`
        unit to convert to when writing output, or None for no conversion

    """

    def __init__(self, default, description="", unit=None):
        self.default = default
        self.description = description
        self.unit = unit

    def __repr__(self):
        return ("Item(default={}, desc='{}', unit={})"
                .format(self.default, self.description, self.unit))
