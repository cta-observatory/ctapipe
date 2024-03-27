import logging
import warnings
from collections import defaultdict
from functools import partial
from inspect import isclass
from pprint import pformat
from textwrap import dedent, wrap

import numpy as np
from astropy.units import Quantity, Unit, UnitConversionError

log = logging.getLogger(__name__)

__all__ = ["Container", "Field", "FieldValidationError", "Map"]


def _fqdn(obj):
    return f"{obj.__module__}.{obj.__qualname__}"


class FieldValidationError(ValueError):
    pass


class Field:
    """
    Class for storing data in a `Container`.

    Parameters
    ----------
    default :
        Default value of the item. This will be set when the `Container`
        is constructed, as well as when  ``Container.reset`` is called.
        This should only be used for immutable values. For mutable values,
        use ``default_factory`` instead.
    description : str
        Help text associated with the item
    unit : str or astropy.units.core.UnitBase
        unit to convert to when writing output, or None for no conversion
    ucd : str
        universal content descriptor (see Virtual Observatory standards)
    type : type
        expected type of value
    dtype : str or np.dtype
        expected data type of the value, None to ignore in validation.
        Means value is expected to be a numpy array or astropy quantity
    ndim : int or None
        expected dimensionality of the data, for arrays, None to ignore
    allow_none : bool
        if the value of None is given to this Field, skip validation
    max_len : int
        if type is str, max_len is the maximum number of bytes of the utf-8
        encoded string to be used.
    default_factory : Callable
        A callable providing a fresh instance as default value.
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
        default_factory=None,
    ):
        self.default = default
        self.default_factory = default_factory
        self.description = description
        self.unit = Unit(unit) if unit is not None else None
        self.ucd = ucd
        self.dtype = np.dtype(dtype) if dtype is not None else None
        self.type = type
        self.ndim = ndim
        self.allow_none = allow_none
        self.max_length = max_length

        if default_factory is not None and default is not None:
            raise ValueError("Must only provide one of default or default_factory")

    def __repr__(self):
        if self.default_factory is not None:
            if isclass(self.default_factory):
                default = _fqdn(self.default_factory)
            elif isinstance(self.default_factory, partial):
                # case for `partial(Map, Container)`
                cls = _fqdn(self.default_factory.args[0])
                if self.default_factory.func is Map:
                    func = "Map"
                else:
                    func = repr(self.default_factory.func)
                default = f"{func}({cls})"
            else:
                # make sure numpy arrays are not dominating everything
                with np.printoptions(threshold=4, precision=3, edgeitems=2):
                    default = str(self.default_factory())
        else:
            default = str(self.default)
        cmps = [f"Field(default={default}"]
        if self.unit is not None:
            cmps.append(f", unit={self.unit}")
        if self.dtype is not None:
            cmps.append(f", dtype={self.dtype}")
        if self.ndim is not None:
            cmps.append(f", ndim={self.ndim}")
        if self.type is not None:
            cmps.append(f", type={self.type.__name__}")
        if self.allow_none is False:
            cmps.append(", allow_none=False")
        if self.max_length is not None:
            cmps.append(f", max_length={self.max_length}")
        cmps.append(")")
        return "".join(cmps)

    def __str__(self):
        desc = f"{self.description} with default {self.default}"
        if self.unit is not None:
            desc += f" [{self.unit}]"
        if self.ndim is not None:
            desc += f" as a {self.ndim}-D array"
        if self.dtype is not None:
            desc += f" with dtype {self.dtype}"
        if self.type is not None:
            desc += f" with type {self.type}"

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

        if isinstance(value, Container):
            # recursively check sub-containers
            value.validate()
            return

        if isinstance(value, Map):
            for key, map_value in value.items():
                if isinstance(map_value, Container):
                    try:
                        map_value.validate()
                    except FieldValidationError as err:
                        raise FieldValidationError(f"[{key}]: {err} ")
            return

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
    """used to mark which fields may be removed in next version"""

    def __init__(self, default, description="", unit=None, ucd=None, reason=""):
        super().__init__(default=default, description=description, unit=unit, ucd=ucd)
        warnings.warn(f"Field {self} is deprecated. {reason}", DeprecationWarning)
        self.reason = reason


_doc_template = """{doc}

Attributes
----------
{fields}
meta : dict
    dict of attached metadata
prefix : str
    Prefix attached to column names when saved to a table or file
"""


def _build_docstring(doc, fields):
    fields = [f"{k} : {f!r}\n    {f.description}" for k, f in fields.items()]
    return _doc_template.format(doc=dedent(doc), fields="\n".join(fields))


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

        for field_name, field in dct["fields"].items():
            field.name = field_name

        dct["__doc__"] = _build_docstring(dct.get("__doc__", ""), dct["fields"])

        new_cls = type.__new__(cls, name, bases, dct)

        # if prefix was not set as a class variable, build a default one
        if "default_prefix" not in dct:
            new_cls.default_prefix = name.lower().replace("container", "")

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
    parent Field is prepended).

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

    def __init__(self, prefix=None, **fields):
        self.meta = {}
        # __slots__ cannot be provided with defaults
        # via class variables, so we use a `default_prefix` class variable
        # and an instance variable `prefix` in `__slots__`
        self.prefix = prefix if prefix is not None else self.default_prefix

        for k in set(self.fields).difference(fields):
            # deepcopy of None is surprisingly slow
            field = self.fields[k]
            if field.default_factory is not None:
                default = field.default_factory()
            else:
                default = field.default
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

    def as_dict(self, recursive=False, flatten=False, add_prefix=False, add_key=False):
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
        add_key: bool
            include map key
        """
        if not recursive:
            return dict(self.items(add_prefix=add_prefix))

        kwargs = dict(
            recursive=recursive,
            add_prefix=add_prefix,
            flatten=flatten,
            add_key=add_key,
        )

        d = dict()
        for key, val in self.items(add_prefix=add_prefix):
            if isinstance(val, Container | Map):
                if flatten:
                    d.update(val.as_dict(**kwargs))
                else:
                    d[key] = val.as_dict(**kwargs)
            else:
                d[key] = val
        return d

    def reset(self):
        """
        Reset all values back to their default values

        Parameters
        ----------
        recursive: bool
            If true, also reset all sub-containers
        """

        for name, field in self.fields.items():
            if field.default_factory is not None:
                setattr(self, name, field.default_factory())
            else:
                setattr(self, name, field.default)

    def update(self, **values):
        """
        update more than one parameter at once (e.g. ``update(x=3,y=4)``
        or ``update(**dict_of_values)``)
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
            desc = "{:>30s}: {}".format(name + extra, str(item))
            lines = wrap(desc, 80, subsequent_indent=" " * 32)
            text.extend(lines)
        return "\n".join(text)

    def validate(self):
        """
        Check that all fields in the Container have the expected characteristics (as
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

    def as_dict(self, recursive=False, flatten=False, add_prefix=False, add_key=False):
        if not recursive:
            return dict(self.items())
        else:
            d = dict()
            for key, val in self.items():
                if isinstance(val, Container) or isinstance(val, Map):
                    if flatten:
                        d.update(
                            {
                                f"{key}_{k}" if add_key else k: v
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

    def __repr__(self):
        if isclass(self.default_factory):
            default = _fqdn(self.default_factory)
        else:
            default = repr(self.default_factory)
        return f"{self.__class__.__name__}({default}, {dict.__repr__(self)!s})"
