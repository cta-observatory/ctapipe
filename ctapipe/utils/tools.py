'''some utils for Tool Developers
'''
from ctapipe.core import non_abstract_children
from traitlets import CaselessStrEnum


def enum_trait(base_class, default, help_str=None):
    '''create a configurable CaselessStrEnum traitlet from baseclass

    the enumeration should contain all names of non_abstract_children()
    of said baseclass and the default choice should be given by
    `base_class._default` name.

    default must be specified and must be the name of one child-class
    '''
    if help_str is None:
        help_str = '{} to use.'.format(base_class.__name__)

    choices = [
        cls.__name__
        for cls in non_abstract_children(base_class)
    ]
    if default not in choices:
        raise ValueError(
            '{default} is not in choices: {choices}'.format(
                default=default,
                choices=choices,
            )
        )

    return CaselessStrEnum(
        choices,
        default,
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
