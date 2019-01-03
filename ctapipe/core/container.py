from collections import defaultdict
from copy import deepcopy
from pprint import pformat
from textwrap import wrap


class Field:
    """
    Class for storing data in `Containers`.

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
        self.ucd = ucd

    def __repr__(self):
        desc = '{}'.format(self.description)
        if self.unit is not None:
            desc += ' [{}]'.format(self.unit)
        return desc


class ContainerMeta(type):
    '''
    The MetaClass for the Containers

    It reserves __slots__ for every class variable,
    that is of instance `Field` and sets all other class variables
    as read-only for the instances.

    This makes sure, that the metadata is immutable,
    and no new fields can be added to a container by accident.
    '''
    def __new__(cls, name, bases, dct):
        field_names = [
            k for k, v in dct.items()
            if isinstance(v, Field)
        ]
        dct['__slots__'] = tuple(field_names + ['meta', 'prefix'])
        dct['fields'] = {}

        # inherit fields from baseclasses
        for b in bases:
            if issubclass(b, Container):
                for k, v in b.fields.items():
                    dct['fields'][k] = v

        for k in field_names:
            dct['fields'][k] = dct.pop(k)

        new_cls = type.__new__(cls, name, bases, dct)

        # if prefix was not set as a class variable, build a default one
        if 'container_prefix' not in dct:
            new_cls.container_prefix = name.lower().replace('container', '')

        return new_cls


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
    output table for example, where each Field defines a column. The
    `dict` conversion can be made recursively and even flattened so
    that a nested set of `Containers` can be translated into a set of
    columns in a flat table without naming conflicts (the name of the
    parent Field is pre-pended).

    Only members of instance `Field` will be used as output.
    For hierarchical data structures, Field can use `Container`
    subclasses or a `Map` as the default value.

    >>>    class MyContainer(Container):
    >>>        x = Field(100,"The X value")
    >>>        energy = Field(-1, "Energy measurement", unit=u.TeV)
    >>>
    >>>    cont = MyContainer()
    >>>    print(cont.x)
    >>>    # metdata will become header keywords in an output file:
    >>>    cont.meta['KEY'] = value

    `Field`s inside `Containers` can contain instances of other
    `Containers`, to allow for a hierarchy of containers, and can also
    contain a `Map` for the case where one wants e.g. a set of
    sub-classes indexed by a value like the `telescope_id`. Examples
    of this can be found in `ctapipe.io.containers`

    `Containers` work by shadowing all class variables (which must be
    instances of `Field`) with instance variables of the same name the
    hold the value expected. If `Container.reset()` is called, all
    instance variables are reset to their default values as defined in
    the class.

    Finally, `Containers` can have associated metadata via their
    `meta` attribute, which is a `dict` of keywords to values.

    """
    def __init__(self, **fields):
        self.meta = {}
        # __slots__ cannot be provided with defaults
        # via class variables, so we use a `__prefix` class variable
        # and a `_prefix` in `__slots__` together with a property.
        self.prefix = self.container_prefix

        for k, v in self.fields.items():
            setattr(self, k, deepcopy(v.default))

        for k, v in fields.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def items(self, add_prefix=False):
        """Generator over (key, value) pairs for the items"""
        if not add_prefix or self.prefix == '':
            return ((k, getattr(self, k)) for k in self.fields.keys())

        return ((self.prefix + '_' + k, getattr(self, k)) for k in self.fields.keys())

    def keys(self):
        """Get the keys of the container"""
        return self.fields.keys()

    def values(self):
        """Get the keys of the container"""
        return (getattr(self, k) for k in self.fields.keys())

    def as_dict(self, recursive=False, flatten=False, add_prefix=False):
        """
        convert the `Container` into a dictionary

        Parameters
        ----------
        recursive: bool
            sub-Containers should also be converted to dicts
        flatten: type
            return a flat dictionary, with any sub-field keys generated
            by appending the sub-Container name.
        """
        if not recursive:
            return dict(self.items(add_prefix=add_prefix))
        else:
            d = dict()
            for key, val in self.items(add_prefix=add_prefix):
                if isinstance(val, Container) or isinstance(val, Map):
                    if flatten:
                        d.update({
                            "{}_{}".format(key, k): v
                            for k, v in val.as_dict(
                                recursive,
                                add_prefix=add_prefix
                            ).items()
                        })
                    else:
                        d[key] = val.as_dict(
                            recursive=recursive, flatten=flatten, add_prefix=add_prefix
                        )
                else:
                    d[key] = val
            return d

    def reset(self, recursive=True):
        """ set all values back to their default values"""
        for name, value in self.fields.items():
            if isinstance(value, Container):
                if recursive:
                    getattr(self, name).reset()
            else:
                setattr(self, name, deepcopy(self.fields[name].default))

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
        for name, item in self.fields.items():
            extra = ""
            if isinstance(getattr(self, name), Container):
                extra = ".*"
            if isinstance(getattr(self, name), Map):
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

    def as_dict(self, recursive=False, flatten=False, add_prefix=False):
        if not recursive:
            return dict(self.items())
        else:
            d = dict()
            for key, val in self.items():
                if isinstance(val, Container) or isinstance(val, Map):
                    if flatten:
                        d.update({
                            "{}_{}".format(key, k): v
                            for k, v in val.as_dict(
                                recursive, add_prefix=add_prefix
                            ).items()
                        })
                    else:
                        d[key] = val.as_dict(
                            recursive=recursive,
                            flatten=flatten,
                            add_prefix=add_prefix,
                        )
                    continue
                d[key] = val
            return d

    def reset(self, recursive=True):
        for val in self.values():
            if isinstance(val, Container):
                val.reset(recursive=recursive)
