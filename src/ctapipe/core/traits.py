"""
Traitlet implementations for ctapipe
"""
import os
import pathlib
from urllib.parse import urlparse

import astropy.units as u
import traitlets
import traitlets.config
from astropy.time import Time
from traitlets import Undefined

from ctapipe.core.plugins import detect_and_import_plugins

from .component import Component, non_abstract_children
from .telescope_component import TelescopeParameter

__all__ = [
    # Implemented here
    "AstroQuantity",
    "AstroTime",
    "BoolTelescopeParameter",
    "IntTelescopeParameter",
    "FloatTelescopeParameter",
    "classes_with_traits",
    "create_class_enum_trait",
    "has_traits",
    # imported from traitlets
    "Path",
    "Bool",
    "CRegExp",
    "CaselessStrEnum",
    "CInt",
    "Dict",
    "Enum",
    "Float",
    "Int",
    "Integer",
    "List",
    "Long",
    "Set",
    "TraitError",
    "Tuple",
    "Unicode",
    "flag",
    "observe",
]

import logging

logger = logging.getLogger(__name__)


class PathError:
    """Signal Non-Existence of a Path"""

    def __init__(self, path, reason):
        self.path = path
        self.reason = reason

    def __repr__(self):
        return f"'{self.path}': {self.reason}"


# Aliases
Bool = traitlets.Bool
Int = traitlets.Int
CInt = traitlets.CInt
Integer = traitlets.Integer
Float = traitlets.Float
Long = traitlets.Long
Unicode = traitlets.Unicode
Dict = traitlets.Dict
Enum = traitlets.Enum
List = traitlets.List
Set = traitlets.Set
CRegExp = traitlets.CRegExp
CaselessStrEnum = traitlets.CaselessStrEnum
UseEnum = traitlets.UseEnum
TraitError = traitlets.TraitError
TraitType = traitlets.TraitType
Tuple = traitlets.Tuple
observe = traitlets.observe
flag = traitlets.config.boolean_flag


class AstroQuantity(TraitType):
    """A trait containing an ``astropy.units`` quantity."""

    def __init__(self, physical_type=None, **kwargs):
        super().__init__(**kwargs)
        if physical_type is not None:
            if isinstance(physical_type, u.PhysicalType):
                self.physical_type = physical_type
            elif isinstance(physical_type, u.UnitBase):
                self.physical_type = u.get_physical_type(physical_type)
            else:
                raise TraitError(
                    "Given physical type must be either of type"
                    " astropy.units.PhysicalType or a subclass of"
                    f" astropy.units.UnitBase, was {type(physical_type)}."
                )
        else:
            self.physical_type = physical_type

        if self.default_value is not Undefined and self.physical_type is not None:
            default_type = u.get_physical_type(self.default_value)
            if default_type != self.physical_type:
                raise TraitError(
                    f"Given physical type {self.physical_type} does not match"
                    f" physical type of the default value, {default_type}."
                )

    def info(self):
        info = "An ``astropy.units.Quantity`` instance"
        if self.allow_none:
            info += "or None"
        return info

    def validate(self, obj, value):
        try:
            quantity = u.Quantity(value)
        except TypeError:
            self.error(obj, value)
        except ValueError:
            self.error(obj, value)

        if self.physical_type is not None:
            given_type = u.get_physical_type(quantity)
            if given_type != self.physical_type:
                raise TraitError(
                    f"Given quantity is of physical type {given_type}."
                    f" Expected {self.physical_type}."
                )

        return quantity


class AstroTime(TraitType):
    """A trait representing a point in Time, as understood by ``astropy.time``."""

    def validate(self, obj, value):
        """try to parse and return an ISO time string"""
        try:
            the_time = Time(value)
            the_time.format = "iso"
            return the_time
        except ValueError:
            self.error(obj, value)

    def info(self):
        info = "an ISO8601 datestring or Time instance"
        if self.allow_none:
            info += "or None"
        return info


class Path(TraitType):
    """
    A path Trait for input/output files.

    Attributes
    ----------
    exists: boolean or None
        If True, path must exist, if False path must not exist
    directory_ok: boolean
        If False, path must not be a directory
    file_ok: boolean
        If False, path must not be a file
    """

    def __init__(
        self,
        default_value=Undefined,
        exists=None,
        directory_ok=True,
        file_ok=True,
        **kwargs,
    ):
        super().__init__(default_value=default_value, **kwargs)
        self.exists = exists
        self.directory_ok = directory_ok
        self.file_ok = file_ok

    def info(self):
        info = "a pathlib.Path or non-empty str for "
        if self.exists is True:
            info += "an existing"
        elif self.exists is False:
            info += "a not existing"
        else:
            info += "a"

        if self.directory_ok and self.file_ok:
            info += " directory or file"
        else:
            if self.file_ok:
                info += " file"
            if self.directory_ok:
                info += "directory"
        if self.allow_none:
            info += " or None"

        return info

    def validate(self, obj, value):
        if isinstance(value, bytes):
            value = os.fsdecode(value)

        if value is None or value is Undefined:
            if self.allow_none:
                return value
            else:
                self.error(obj, value)

        if not isinstance(value, str | pathlib.Path):
            self.error(obj, value)

        # expand any environment variables in the path:
        value = os.path.expandvars(value)

        if isinstance(value, str):
            if value == "":
                self.error(obj, value)

            try:
                url = urlparse(value)
            except ValueError:
                self.error(obj, value)

            if url.scheme in ("http", "https"):
                # here to avoid circular import, since every module imports
                # from ctapipe.core
                from ctapipe.utils.download import download_cached

                value = download_cached(value, progress=True)
            elif url.scheme == "dataset":
                # here to avoid circular import, since every module imports
                # from ctapipe.core
                from ctapipe.utils import get_dataset_path

                value = get_dataset_path(value.partition("dataset://")[2])
            elif url.scheme in ("", "file"):
                value = pathlib.Path(url.netloc, url.path)
            else:
                self.error(obj, value)

        value = value.absolute()
        exists = value.exists()
        if self.exists is not None:
            if exists != self.exists:
                raise TraitError(PathError(value, "does not exist"), self.info(), self)
        if exists:
            if not self.directory_ok and value.is_dir():
                raise TraitError(PathError(value, "is a directory"), self.info(), self)
            if not self.file_ok and value.is_file():
                raise TraitError(PathError(value, "is a file"), self.info(), self)
        return value


def create_class_enum_trait(base_class, default_value, help=None, allow_none=False):
    """create a configurable CaselessStrEnum traitlet from baseclass

    the enumeration should contain all names of non_abstract_children()
    of said baseclass and the default choice should be given by
    ``base_class._default`` name.

    default must be specified and must be the name of one child-class
    """
    if help is None:
        help = "{} to use.".format(base_class.__name__)

    choices = [cls.__name__ for cls in non_abstract_children(base_class)]

    if default_value not in choices:
        raise ValueError(f"{default_value} is not in choices: {choices}")

    return CaselessStrEnum(
        choices,
        default_value=default_value,
        help=help,
        allow_none=allow_none,
    ).tag(config=True)


class ComponentName(Unicode):
    """A trait that is the name of a Component class"""

    def __init__(self, cls, **kwargs):
        # we need to prevent triggering importing plugins at
        # import time to avoid circular imports, this flag is used
        # to prevent calling the full plugin mechanism at definition
        # time of a `ComponentName`
        self._init_done = False

        if not issubclass(cls, Component):
            raise TypeError(f"cls must be a Component, got {cls}")

        self.cls = cls
        super().__init__(**kwargs)
        if "help" not in kwargs:
            self.help = f"The name of a {cls.__name__} subclass"
        self._init_done = True

    @property
    def help(self):
        if self._init_done:
            children = list(self.cls.non_abstract_subclasses())
        else:
            children = []
        return f"{self._help}. Possible values: {children}"

    @help.setter
    def help(self, value):
        self._help = value

    @property
    def info_text(self):
        if self._init_done:
            return f"Any of {list(self.cls.non_abstract_subclasses())}"
        else:
            return f"Any subclass of {self.cls}"

    def validate(self, obj, value):
        if self.allow_none and value is None:
            return None

        if value in self.cls.non_abstract_subclasses():
            return value

        self.error(obj, value)


class ComponentNameList(List):
    """A trait that is a list of Component classes"""

    def __init__(self, cls, **kwargs):
        # we need to prevent triggering importing plugins at
        # import time to avoid circular imports, this flag is used
        # to prevent calling the full plugin mechanism at definition
        # time of a `ComponentNameList`
        self._init_done = False
        if not issubclass(cls, Component):
            raise TypeError(f"cls must be a Component, got {cls}")

        self.cls = cls
        trait = ComponentName(cls)
        super().__init__(trait=trait, **kwargs)

        if "help" not in kwargs:
            self.help = f"A list of {cls.__name__} subclass names"
        self._init_done = True

    @property
    def help(self):
        if self._init_done:
            children = list(self.cls.non_abstract_subclasses())
        else:
            children = []
        return f"{self._help}. Possible values: {children}"

    @help.setter
    def help(self, value):
        self._help = value

    @property
    def info_text(self):
        if self._init_done:
            return f"A list of {list(self.cls.non_abstract_subclasses())}"
        else:
            return f"A list of {self.cls} subclasses"


def classes_with_traits(base_class):
    """Returns a list of the base class plus its non-abstract children
    if they have traits"""

    if hasattr(base_class, "plugin_entry_point"):
        detect_and_import_plugins(base_class.plugin_entry_point)

    all_classes = [base_class] + non_abstract_children(base_class)
    with_traits = []

    for cls in all_classes:
        if has_traits(cls):
            with_traits.append(cls)

        # add subcomponents
        if hasattr(cls, "classes"):
            # we will ignore failing classes to not break anyone
            if isinstance(cls.classes, List):
                classes = cls.classes.default()
            else:
                classes = cls.classes

            try:
                for component in classes:
                    with_traits.extend(classes_with_traits(component))
            except Exception:
                pass

    return with_traits


def has_traits(cls, ignore=("config", "parent")):
    """True if cls has any traits apart from the usual ones

    all our components have at least 'config' and 'parent' as traitlets
    this is inherited from `traitlets.config.Configurable` so we ignore them
    here.
    """
    return bool(set(cls.class_trait_names()) - set(ignore))


class FloatTelescopeParameter(TelescopeParameter):
    """a `~ctapipe.core.telescope_component.TelescopeParameter` with Float trait type"""

    def __init__(self, **kwargs):
        """Create a new FloatTelescopeParameter"""
        super().__init__(trait=Float(), **kwargs)


class IntTelescopeParameter(TelescopeParameter):
    """a `~ctapipe.core.telescope_component.TelescopeParameter` with Int trait type"""

    def __init__(self, **kwargs):
        """Create a new IntTelescopeParameter"""
        super().__init__(trait=Integer(), **kwargs)


class BoolTelescopeParameter(TelescopeParameter):
    """a `~ctapipe.core.telescope_component.TelescopeParameter` with Bool trait type"""

    def __init__(self, **kwargs):
        """Create a new BoolTelescopeParameter"""
        super().__init__(trait=Bool(), **kwargs)
