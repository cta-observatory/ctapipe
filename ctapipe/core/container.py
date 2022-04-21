from collections import defaultdict
from copy import deepcopy
from pprint import pformat
from textwrap import wrap
import warnings
import numpy as np
from astropy.units import UnitConversionError, Quantity, Unit

import logging


log = logging.getLogger(__name__)

__all__ = ["Container", "Field", "FieldValidationError", "Map"]


class FieldValidationError(ValueError):
    pass


class Field:
    """
    Class for storing data in a `Container`.

    Parameters
    ----------
    default:
        default value of the item (this will be set when the `Container`
        is constructed, as well as when  ``Container.reset`` is called
    description: str
        Help text associated with the item
    unit: str or astropy.units.core.UnitBase
        unit to convert to when writing output, or None for no conversion
    ucd: str
        universal content descriptor (see Virtual Observatory standards)
    type: type
        expected type of value
    dtype: str or np.dtype
        expected data type of the value, None to ignore in validation.
        Means value is expected to be a numpy array or astropy quantity
    ndim: int or None
        expected dimensionality of the data, for arrays, None to ignore
    allow_none:
        if the value of None is given to this Field, skip validation
    max_len:
        if type is str, max_len is the maximum number of bytes of the utf-8
        encoded string to be used.
    """

    def __init__(
        self,
        default=None,
        description="",
        unit=None,
        ucd=None,
        dtype=None,
        type=None,
        ndim=None,
        allow_none=True,
        max_length=None,
    ):

        self.default = default
        self.description = description
        self.unit = Unit(unit) if unit is not None else None
        self.ucd = ucd
        self.dtype = np.dtype(dtype) if dtype is not None else None
        self.type = type
        self.ndim = ndim
        self.allow_none = allow_none
        self.max_length = max_length

    def __repr__(self):
        desc = f"{self.description}"
        if self.unit is not None:
            desc += f" [{self.unit}]"
        if self.ndim is not None:
            desc += f" as a {self.ndim}-D array"
        if self.dtype is not None:
            desc += f" with type {self.dtype}"

        return desc

    def validate(self, value):
        """
        check that a given value is appropriate for this Field

        Parameters
        ----------
        value: Any
           the value to test

        Raises
        ------
        FieldValidationError:
            if the value is not valid
        """

        if self.allow_none and value is None:
            return

        errorstr = f"the value '{value}' ({type(value)}) is invalid: "

        if self.type is not None and not isinstance(value, self.type):
            raise FieldValidationError(
                f"{errorstr} Should be an instance of {self.type}"
            )

        if self.unit is not None:
            if not isinstance(value, Quantity):
                raise FieldValidationError(
                    f"{errorstr} Should have units of {self.unit}"
                ) from None
            try:
                value.to(self.unit)
            except UnitConversionError as err:
                raise FieldValidationError(f"{errorstr}: {err}")

            # strip off the units now, so we can test the rest without units
            value = value.value

        if self.ndim is not None:
            # should be a numpy array
            if not isinstance(value, np.ndarray):
                raise FieldValidationError(f"{errorstr} Should be an ndarray")
            if value.ndim != self.ndim:
                raise FieldValidationError(
                    f"{errorstr} Should have dimensionality {self.ndim}"
                )
            if value.dtype != self.dtype:
                raise FieldValidationError(
                    f"{errorstr} Has dtype "
                    f"{value.dtype}, should have dtype"
                    f" {self.dtype}"
                )
        else:
            # not a numpy array
            if self.dtype is not None:
                if not isinstance(value, self.dtype.type):
                    raise FieldValidationError(
                        f"{errorstr} Should have numpy dtype {self.dtype}"
                    )


class DeprecatedField(Field):
    """ used to mark which fields may be removed in next version """

    def __init__(self, default, description="", unit=None, ucd=None, reason=""):
        super().__init__(default=default, description=description, unit=unit, ucd=ucd)
        warnings.warn(f"Field {self} is deprecated. {reason}", DeprecationWarning)
        self.reason = reason


class ContainerMeta(type):
    """
    The MetaClass for `Container`

    It reserves __slots__ for every class variable,
    that is of instance `Field` and sets all other class variables
    as read-only for the instances.

    This makes sure, that the metadata is immutable,
    and no new fields can be added to a container by accident.
    """

    def __new__(cls, name, bases, dct):
        field_names = [k for k, v in dct.items() if isinstance(v, Field)]
        dct["__slots__"] = tuple(field_names + ["meta", "prefix"])
        dct["fields"] = {}

        # inherit fields from baseclasses
        for b in bases:
            if issubclass(b, Container):
                for k, v in b.fields.items():
                    dct["fields"][k] = v

        for k in field_names:
            dct["fields"][k] = dct.pop(k)

        new_cls = type.__new__(cls, name, bases, dct)

        # if prefix was not set as a class variable, build a default one
        if "container_prefix" not in dct:
            new_cls.container_prefix = name.lower().replace("container", "")

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

    Containers can transform the data into a `dict` using the
    ``as_dict`` method.  This allows them to be written to an
    output table for example, where each Field defines a column. The
    `dict` conversion can be made recursively and even flattened so
    that a nested set of `Containers <Container>`_ can be translated into a set of
    columns in a flat table without naming conflicts (the name of the
    parent Field is pre-pended).

    Only members of instance `Field` will be used as output.
    For hierarchical data structures, Field can use `Container`
    subclasses or a `Map` as the default value.

    >>> import astropy.units as u
    >>> class MyContainer(Container):
    ...     x = Field(100, "The X value")
    ...     energy = Field(-1, "Energy measurement", unit=u.TeV)
    ...
    >>> cont = MyContainer()
    >>> print(cont.x)
    100
    >>> # metadata will become header keywords in an output file:
    >>> cont.meta["KEY"] = "value"

    `Fields <Field>`_ inside `Containers <Container>`_ can contain instances of other
    containers, to allow for a hierarchy of containers, and can also
    contain a `Map` for the case where one wants e.g. a set of
    sub-classes indexed by a value like the ``telescope_id``. Examples
    of this can be found in `ctapipe.containers`

    `Container` works by shadowing all class variables (which must be
    instances of `Field`) with instance variables of the same name that
    hold the actual data. If ``reset`` is called, all
    instance variables are reset to their default values as defined in
    the class.

    Finally, a Container can have associated metadata via its
    `meta` attribute, which is a `dict` of keywords to values.

    """

    def __init__(self, **fields):
        self.meta = {}
        # __slots__ cannot be provided with defaults
        # via class variables, so we use a `container_prefix` class variable
        # and an instance variable `prefix` in `__slots__`
        self.prefix = self.container_prefix

        for k in set(self.fields).difference(fields):

            # deepcopy of None is surprisingly slow
            default = self.fields[k].default
            if default is not None:
                default = deepcopy(default)

            setattr(self, k, default)

        for k, v in fields.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def items(self, add_prefix=False):
        """Generator over (key, value) pairs for the items"""
        if not add_prefix or self.prefix == "":
            return ((k, getattr(self, k)) for k in self.fields.keys())

        return ((self.prefix + "_" + k, getattr(self, k)) for k in self.fields.keys())

    def keys(self):
        """Get the keys of the container"""
        return self.fields.keys()

    def values(self):
        """Get the keys of the container"""
        return (getattr(self, k) for k in self.fields.keys())

    def as_dict(self, recursive=False, flatten=False, add_prefix=False):
        """
        Convert the `Container` into a dictionary

        Parameters
        ----------
        recursive: bool
            sub-Containers should also be converted to dicts
        flatten: type
            return a flat dictionary, with any sub-field keys generated
            by appending the sub-Container name.
        add_prefix: bool
            include the container's prefix in the name of each item
        """
        if not recursive:
            return dict(self.items(add_prefix=add_prefix))
        else:
            d = dict()
            for key, val in self.items(add_prefix=add_prefix):
                if isinstance(val, Container) or isinstance(val, Map):
                    if flatten:
                        d.update(
                            {
                                f"{key}_{k}": v
                                for k, v in val.as_dict(
                                    recursive, add_prefix=add_prefix
                                ).items()
                            }
                        )
                    else:
                        d[key] = val.as_dict(
                            recursive=recursive, flatten=flatten, add_prefix=add_prefix
                        )
                else:
                    d[key] = val
            return d

    def reset(self, recursive=True):
        """
        Reset all values back to their default values

        Parameters
        ----------
        recursive: bool
            If true, also reset all sub-containers
        """

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
            lines = wrap(desc, 80, subsequent_indent=" " * 32)
            text.extend(lines)
        return "\n".join(text)

    def validate(self):
        """
        Check that all fields in the Container have the expected characterisics (as
        defined by the Field metadata).  This is not intended to be run every time a
        Container is filled, since it is slow, only for testing a first event.

        Raises
        ------
        ValueError:
            if the Container's values are not valid
        """
        for name, field in self.fields.items():
            try:
                field.validate(self[name])
            except FieldValidationError as err:
                raise FieldValidationError(
                    f"{self.__class__.__name__} Field '{name}': {err}"
                )


class Map(defaultdict):
    """A dictionary of sub-containers that can be added to a Container. This
    may be used e.g. to store a set of identical sub-Containers (e.g. indexed
    by ``tel_id`` or algorithm name).
    """

    def as_dict(self, recursive=False, flatten=False, add_prefix=False):
        if not recursive:
            return dict(self.items())
        else:
            d = dict()
            for key, val in self.items():
                if isinstance(val, Container) or isinstance(val, Map):
                    if flatten:
                        d.update(
                            {
                                f"{key}_{k}": v
                                for k, v in val.as_dict(
                                    recursive, add_prefix=add_prefix
                                ).items()
                            }
                        )
                    else:
                        d[key] = val.as_dict(
                            recursive=recursive, flatten=flatten, add_prefix=add_prefix
                        )
                    continue
                d[key] = val
            return d

    def reset(self, recursive=True):
        """Calls all ``Container.reset`` for all values in the Map"""
        for val in self.values():
            if isinstance(val, Container):
                val.reset(recursive=recursive)
