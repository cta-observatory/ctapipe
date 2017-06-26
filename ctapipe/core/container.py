from collections import defaultdict
from copy import deepcopy
from pprint import pformat
from textwrap import wrap


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
    ucd: str
        universal content descriptor (see Virtual Observatory standards)
    """

    def __init__(self, default, description="", unit=None, ucd=None):
        self.default = default
        self.description = description
        self.unit = unit

    def __repr__(self):
        desc = '{}'.format(self.description)
        if self.unit is not None:
            desc += ' [{}]'.format(self.unit)
        return desc


class ContainerMeta(type):
    '''
    The MetaClass for the Containers

    It reserves __slots__ for every class variable,
    that is of instance `Item` and sets all other class variables
    as read-only for the instances.

    This makes sure, that the metadata is immutable,
    and no new fields can be added to a container by accident.
    '''
    def __new__(cls, name, bases, dct):
        items = [
            k for k, v in dct.items()
            if isinstance(v, Item)
        ]
        dct['__slots__'] = tuple(items + ['meta'])
        dct['_items'] = {}

        for k in items:
            dct['_items'][k] = dct.pop(k)

        return type.__new__(cls, name, bases, dct)


class Container(metaclass=ContainerMeta):
    """Generic class that can hold and accumulate data to be passed
    between Components.

    The purpose of this class is to provide a flexible data structure
    that works a bit like a dict or blank Python class, but prevents
    the user from accessing members that have not been defined a
    priori (more like a C struct), and also keeps metadata information
    such as a description, defaults, and units for each item in the
    container.

    Containers can transform the data into a `dict` using the `
    Container.as_dict()` method.  This allows them to be written to an
    output table for example, where each Item defines a column. The
    `dict` conversion can be made recursively and even flattened so
    that a nested set of `Containers` can be translated into a set of
    columns in a flat table without naming conflicts (the name of the
    parent Item is pre-pended).

    Only members of instance `Item` will be used as output.
    For hierarchical data structures, Items can use `Container`
    subclasses or a `Map` as the default value.

    You should not make class hierarchies of Containers and only ever
    subclass the Container base class

    >>>    class MyContainer(Container):
    >>>        x = Item(100,"The X value")
    >>>        energy = Item(-1, "Energy measurement", unit=u.TeV)
    >>>
    >>>    cont = MyContainer()
    >>>    print(cont.x)
    100
    >>>    # metadata will become header keywords in an output file:
    >>>    cont.meta['KEY'] = value

    `Items` inside `Containers` can contain instances of other
    `Containers`, to allow for a hierarchy of containers, and can also
    contain a `Map` for the case where one wants e.g. a set of
    sub-classes indexed by a value like the `telescope_id`. Examples
    of this can be found in `ctapipe.io.containers`

    Finally, `Containers` can have associated metadata via their
    `meta` attribute, which is a `dict` of keywords to values.

    """

    def __init__(self, **items):

        self.meta = {}
        for k, v in self._items.items():
            setattr(self, k, deepcopy(v.default))
            self.meta[k] = v

        for k, v in items.items():
            setattr(self, k, v)

    def __getitem__(self, item):
        if item in self.__slots__:
            return getattr(self, item)
        else:
            raise KeyError('No such Item: "{}"'.format(item))

    def __setitem__(self, item, value):
        if item in self.__slots__:
            return setattr(self, item, value)
        else:
            raise KeyError('No such Item: "{}"'.format(item))

    @property
    def attributes(self):
        """
        a dict of the Item metadata of each attribute.
        """
        return self.meta

    def items(self):
        """Generator over (key, value) pairs for the items"""
        return ((k, getattr(self, k)) for k in self._items.keys())

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
            return dict(self.items())
        else:
            d = dict()
            for key, val in self.items():
                if key.startswith("_"):
                    continue
                if isinstance(val, Container) or isinstance(val, Map):
                    if flatten:
                        d.update({
                            "{}_{}".format(key, k): v
                            for k, v in val.as_dict(recursive).items()
                        })
                    else:
                        d[key] = val.as_dict(recursive=recursive,
                                             flatten=flatten)
                    continue
                d[key] = val
            return d

    def reset(self, recursive=True):
        """ set all values back to their default values"""
        for name, value in self.meta.items():
            if isinstance(value, Container):
                if recursive:
                    getattr(self, name).reset()
            else:
                setattr(self, name, deepcopy(self.meta[name].default))

    def update(self, **values):
        """
        update more than one parameter at once (e.g. `update(x=3,y=4)`
        or `update(**dict_of_values)`)
        """
        for key in values:
            self[key] = values[key]

    def __str__(self):
        return pformat(self.as_dict(recursive=True))

    def __repr__(self):
        text = ["{}.{}:".format(type(self).__module__, type(self).__name__)]
        for name, item in self._items.items():
            extra = ""
            if isinstance(item, Container):
                extra = ".*"
            if isinstance(item, Map):
                extra = "[*]"
            desc = "{:>30s}: {}".format(name+extra, repr(item))
            lines = wrap(desc, 80, subsequent_indent=' '*32)
            text.extend(lines)
        return "\n".join(text)

    def __getstate__(self):
        state = dict(self.items())
        state['meta'] = getattr(self, 'meta')
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)


class Map(defaultdict):
    """A dictionary of sub-containers that can be added to a Container. This
    may be used e.g. to store a set of identical sub-Containers (e.g. indexed
    by `tel_id` or algorithm name).
    """

    def as_dict(self, recursive=False, flatten=False):
        if not recursive:
            return dict(self.items())
        else:
            d = dict()
            for key, val in self.items():
                if isinstance(val, Container) or isinstance(val, Map):
                    if flatten:
                        d.update({"{}_{}".format(key, k): v
                                  for k, v in val.as_dict(recursive).items()})
                    else:
                        d[key] = val.as_dict(recursive=recursive,
                                             flatten=flatten)
                    continue
                d[key] = val
            return d

    def reset(self, recursive=True):
        for key, val in self.items():
            if isinstance(val, Container):
                val.reset(recursive=True)
