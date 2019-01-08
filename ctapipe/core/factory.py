from inspect import isabstract
from traitlets import CaselessStrEnum


def non_abstract_children(base):
    """
    Return all non-abstract subclasses of a base class recursively.
    """
    subclasses = base.__subclasses__() + [
        g for s in base.__subclasses__()
        for g in non_abstract_children(s)
    ]
    non_abstract = [g for g in subclasses if not isabstract(g)]

    return non_abstract


def has_traits(cls, ignore=('config', 'parent')):
    '''True if cls has any traits apart from the usual ones

    all our components have at least 'config' and 'parent' as traitlets
    this is inherited from `traitlets.config.Configurable` so we ignore them
    here.
    '''
    return bool(
        set(cls.class_trait_names()) - set(ignore)
    )


def from_name(cls_name, default, namespace, *args, **kwargs):
    if cls_name is None:
        cls_name = default

    cls = namespace[cls_name]
    return cls(*args, **kwargs)


def enum_trait(base_class, default, help_str):
    return CaselessStrEnum(
        [
            cls.__name__
            for cls in non_abstract_children(base_class)
        ],
        default,
        allow_none=True,
        help=help_str
    ).tag(config=True)


def classes_with_traits(base_class):
    all_classes = [base_class] + non_abstract_children(base_class)
    return [cls for cls in all_classes if has_traits(cls)]
