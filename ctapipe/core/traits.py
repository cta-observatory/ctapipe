import os

from traitlets import (
    Bool,
    CaselessStrEnum,
    CRegExp,
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
    "CRegExp",
    "Dict",
    "flag",
    "TraitError",
    "observe",
    "CaselessStrEnum",
    "enum_trait",
    "classes_with_traits",
    "has_traits",
]


class Path(TraitType):
    def __init__(self, exists=None, directory_ok=True, file_ok=True):
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
