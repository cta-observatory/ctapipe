'''some utils for Tool Developers
'''
from ctapipe.core import non_abstract_children
from traitlets import CaselessStrEnum


def enum_trait(base_class, help_str=None):
    '''create a configurable CaselessStrEnum traitlet from baseclass

    the enumeration should contain all names of non_abstract_children()
    of said baseclass and the default choice should be given by
    `base_class._default` name.

    If the base_class has no `_default_name` an exception is raised, as
    this class is not suitable to be used with this function.
    '''
    if help_str is None:
        help_str = '{} to use.'.format(base_class.__name__)

    return CaselessStrEnum(
        [
            cls.__name__
            for cls in non_abstract_children(base_class)
        ],
        base_class._default_name,
        allow_none=True,
        help=help_str
    ).tag(config=True)


def classes_with_traits(base_class):
    all_classes = [base_class] + non_abstract_children(base_class)
    return [cls for cls in all_classes if has_traits(cls)]


def has_traits(cls, ignore=('config', 'parent')):
    '''True if cls has any traits apart from the usual ones

    all our components have at least 'config' and 'parent' as traitlets
    this is inherited from `traitlets.config.Configurable` so we ignore them
    here.
    '''
    return bool(
        set(cls.class_trait_names()) - set(ignore)
    )
