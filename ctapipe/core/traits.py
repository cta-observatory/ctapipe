import os
from fnmatch import fnmatch

from traitlets import (
    Bool,
    CaselessStrEnum,
    Dict,
    Enum,
    Float,
    Int,
    Integer,
    List,
    Long,
    TraitError,
    TraitType,
    Unicode,
    observe,
    Set,
    CRegExp,
)
from traitlets.config import boolean_flag as flag

from .component import non_abstract_children

__all__ = [
    "Path",
    "Int",
    "Integer",
    "Float",
    "Unicode",
    "Enum",
    "Long",
    "List",
    "Bool",
    "Set",
    "CRegExp",
    "Dict",
    "flag",
    "TraitError",
    "observe",
    "CaselessStrEnum",
    "enum_trait",
    "classes_with_traits",
    "has_traits",
    "TelescopeParameter",
    "FloatTelescopeParameter",
    "IntTelescopeParameter",
    "TelescopeParameterResolver",
]

import logging

logger = logging.getLogger(__name__)


class Path(TraitType):
    """
    A path Trait for input/output files.

    Parameters
    ----------
    exists: boolean or None
        If True, path must exist, if False path must not exist

    directory_ok: boolean
        If False, path must not be a directory
    file_ok: boolean
        If False, path must not be a file
    """

    def __init__(self, exists=None, directory_ok=True, file_ok=True):
        super().__init__()
        self.exists = exists
        self.directory_ok = directory_ok
        self.file_ok = file_ok

    def validate(self, obj, value):

        if isinstance(value, str):
            value = os.path.abspath(value)
            if self.exists is not None:
                if os.path.exists(value) != self.exists:
                    raise TraitError(
                        'Path "{}" {} exist'.format(
                            value, "does not" if self.exists else "must"
                        )
                    )
            if os.path.exists(value):
                if os.path.isdir(value) and not self.directory_ok:
                    raise TraitError(f'Path "{value}" must not be a directory')
                if os.path.isfile(value) and not self.file_ok:
                    raise TraitError(f'Path "{value}" must not be a file')

            return value

        return self.error(obj, value)


def enum_trait(base_class, default, help_str=None):
    """create a configurable CaselessStrEnum traitlet from baseclass

    the enumeration should contain all names of non_abstract_children()
    of said baseclass and the default choice should be given by
    `base_class._default` name.

    default must be specified and must be the name of one child-class
    """
    if help_str is None:
        help_str = "{} to use.".format(base_class.__name__)

    choices = [cls.__name__ for cls in non_abstract_children(base_class)]
    if default not in choices:
        raise ValueError(
            "{default} is not in choices: {choices}".format(
                default=default, choices=choices
            )
        )

    return CaselessStrEnum(choices, default, allow_none=True, help=help_str).tag(
        config=True
    )


def classes_with_traits(base_class):
    """ Returns a list of the base class plus its non-abstract children 
    if they have traits """
    all_classes = [base_class] + non_abstract_children(base_class)
    return [cls for cls in all_classes if has_traits(cls)]


def has_traits(cls, ignore=("config", "parent")):
    """True if cls has any traits apart from the usual ones

    all our components have at least 'config' and 'parent' as traitlets
    this is inherited from `traitlets.config.Configurable` so we ignore them
    here.
    """
    return bool(set(cls.class_trait_names()) - set(ignore))


class TelescopeParameter(List):
    """
    Allow a parameter value to be specified as a simple value (of type *dtype*),
    or as a list of patterns that match different telescopes.
    The patterns are given as a list of 3-tuples in in the
    form: `[(command, argument, value), ...]`.

    Command can be one of:
    - 'type': argument is then a telescope type  string (e.g.
       `('type', 'SST_ASTRI_CHEC', 4.0)` to apply to all telescopes of that type,
       or use a wildcard like "LST*", or "*" to set a pure default value for all
       telescopes.
    - 'id':  argument is a specific telescope ID `['id', 89, 5.0]`)

    These are evaluated in-order, so you can first set a default value, and then set
    values for specific telescopes or types to override them.

    Examples
    --------

    .. code-block: python
    tel_param = [
        ('type', '*', 5.0),                       # default for all
        ('type', 'LST_*', 5.2),
        ('type', 'MST_MST_NectarCam', 4.0),
        ('type', 'MST_MST_FlashCam', 4.5),
        ('id', 34, 4.0),                   # override telescope 34 specifically
    ]

    .. code-block: python
    tel_param = 4.0  # sets this value for all telescopes

    """

    def __init__(self, dtype=float, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(dtype, type):
            raise ValueError("dtype should be a type")
        self._dtype = dtype

    def validate(self, obj, value):
        # support a single value for all (convert into a default value)
        if isinstance(value, self._dtype):
            value = [("type", "*", value)]

        # check that it is a list
        super().validate(obj, value)
        normalized_value = []

        for pattern in value:
            # now check for the standard 3-tuple of )command, argument, value)
            if len(pattern) != 3:
                raise TraitError(
                    "pattern should be a tuple of (command, argument, value)"
                )
            command, arg, val = pattern
            if not isinstance(val, self._dtype):
                raise TraitError(f"Value should be a {self._dtype}")
            if not isinstance(command, str):
                raise TraitError("command must be a string")
            if command not in ["type", "id"]:
                raise TraitError("command must be one of: '*', 'type', 'id'")
            if command == "type":
                if not isinstance(arg, str):
                    raise TraitError("'type' argument should be a string")
            if command == "id":
                arg = int(arg)

            val = self._dtype(val)
            normalized_value.append((command, arg, val))

        return normalized_value


class FloatTelescopeParameter(TelescopeParameter):
    """ a `TelescopeParameter` with float type (see docs for `TelescopeParameter`)"""

    def __init__(self, **kwargs):
        super().__init__(dtype=float, **kwargs)


class IntTelescopeParameter(TelescopeParameter):
    """ a `TelescopeParameter` with int type (see docs for `TelescopeParameter`)"""

    def __init__(self, **kwargs):
        super().__init__(dtype=int, **kwargs)


class TelescopeParameterResolver:
    def __init__(
        self,
        subarray: "ctapipe.instrument.SubarrayDescription",
        tel_param: "TelescopeParameter",
    ):
        """
        Handles looking up a parameter by telescope_id, given a TelescopeParameter
        trait (which maps a parameter to a set of telescopes by type, id, or other
        selection criteria).

        Parameters
        ----------
        name: str
            name of the mapped parameter
        subarray: ctapipe.instrument.SubarrayDescription
            description of the subarray (includes mapping of tel_id to tel_type)
        tel_param: TelescopeParameter trait
            the parameter definitions
        """

        # build dictionary mapping tel_id to parameter:
        self._value_for_tel_id = {}

        for command, argument, value in tel_param:
            if command == "type":
                matched_tel_types = [
                    t for t in subarray.telescope_types if fnmatch(t, argument)
                ]
                logger.debug(f"argument '{argument}' matched: {matched_tel_types}")
                if len(matched_tel_types) == 0:
                    logger.warning(
                        "TelescopeParameter type argument '%s' did not match "
                        "any known telescope types",
                        argument,
                    )
                for tel_type in matched_tel_types:
                    for tel_id in subarray.get_tel_ids_for_type(tel_type):
                        self._value_for_tel_id[tel_id] = value
            elif command == "id":
                self._value_for_tel_id[int(argument)] = value
            else:
                raise ValueError(f"Unrecognized command: {command}")

    def value_for_tel_id(self, tel_id: int):
        """
        returns the resolved parameter for the given telescope id
        """
        try:
            return self._value_for_tel_id[tel_id]
        except KeyError:
            raise KeyError(
                f"TelescopeParameterResolver: no "
                f"parameter value was set for telescope with tel_id="
                f"{tel_id}. Please set it explicitly, or by telescope type or '*'."
            )
