from collections import defaultdict
from copy import copy
from pprint import pformat
from textwrap import wrap


class Container:
    """Generic class that can hold and accumulate data to be passed
    between Components.

    The purpose of this class is to provide a flexible data structure
    that works a bit like a dict or blank Python class, but prevents
    the user from accessing members that have not been defined a
    priori (more like a C struct), and also keeps metdata information
    such as a description, defaults, and units for each item in the
    container.

    Containers can transform the data into a `dict` using the `
    Container.as_dict()` method.  This allows them to be written to an
    output table for example, where each Item defines a column. The
    `dict` conversion can be made recursively and even flattened so
    that a nested set of `Containers` can be translated into a set of
    columns in a flat table without naming conflicts (the name of the
    parent Item is pre-pended).

    To use this class, all members must be defined as `Item`s with
    default values specified.  For hierarchical data structures, Items
    can use `Container` subclasses or a `Map` as the default value.

    You should not make class hierarchies of Containers and only ever
    subclass the Container base class

    >>>    class MyContainer(Container):
    >>>        x = Item(100,"The X value")
    >>>        energy = Item(-1, "Energy measurement", unit=u.TeV)
    >>>
    >>>    cont = MyContainer()
    >>>    print(cont.x)
    100
    >>>    # metdata will become header keywords in an output file:
    >>>    cont.meta['KEY'] = value  

    `Items` inside `Containers` can contain instances of other
    `Containers`, to allow for a hierarchy of containers, and can also
    contain a `Map` for the case where one wants e.g. a set of
    sub-classes indexed by a value like the `telescope_id`. Examples
    of this can be found in `ctapipe.io.containers`

    `Containers` work by shadowing all class variables (which must be
    instances of `Item`) with instance variables of the same name the
    hold the value expected. If `Container.reset()` is called, all
    instance variables are reset to their default values as defined in
    the class.

    Finally, `Containers` can have associated metadata via their
    `meta` attribute, which is a `dict` of keywords to values.

    """

    def __init__(self, **values):
        object.__setattr__(self, "_metadata", dict())
        self.reset()
        for key in values:
            self[key] = values[key]

    def __setattr__(self, name, value):
        """Prevent new attributes that aren't in the class definition"""
        if hasattr(self.__class__, name):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                "{} has no attribute '{}'".format(self.__class__, name))

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        if hasattr(self.__class__, key):
            self.__dict__[key] = value
        else:
            raise AttributeError("{} has no attribute '{}'"
                                 .format(self.__class__, key))

    @property
    def meta(self):
        """metadata key/values associated with this Container.

        When written to an output file, these will become headers, so
        should represent data that does not change after the
        `Container` is constructed.
        """
        return self._metadata

    @property
    def attributes(self):
        """
        a dict of the Item metadata of each attribute.
        """
        return {key: val for key, val in self.__class__.__dict__.items()
                if isinstance(val, Item)}

    def items(self):
        """dict-like access, but skip any hidden items like _metadata"""
        return ((k, v) for k, v in self.__dict__.items()
                if not k.startswith('_'))

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
                        d.update({"{}_{}".format(key, k): v
                                  for k, v in val.as_dict(recursive).items()})
                    else:
                        d[key] = val.as_dict(recursive=recursive,
                                             flatten=flatten)
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
        text = ["{}.{}:".format(type(self).__module__, type(self).__name__), ]
        for name, item in self.attributes.items():
            extra = ""
            if isinstance(self.__dict__[name], Container):
                extra = ".*"
            if isinstance(self.__dict__[name], Map):
                extra = "[*]"
            desc = "{:>30s}: {}".format(name + extra, repr(item))
            lines = wrap(desc, 80, subsequent_indent=' ' * 32)
            text.extend(lines)
        return "\n".join(text)


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
