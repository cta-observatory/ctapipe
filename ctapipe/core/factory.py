from inspect import isabstract


def child_subclasses(base):
    """
    Return all non-abstract subclasses of a base class.

    Parameters
    ----------
    base : class
        high level class object that is inherited by the
        desired subclasses

    Returns
    -------
    children : list
        list of non-abstract subclasses

    """
    family = base.__subclasses__() + [
        g for s in base.__subclasses__()
        for g in child_subclasses(s)
    ]
    children = [g for g in family if not isabstract(g)]

    return children


def has_traits(cls, ignore=('config', 'parent')):
    '''True if cls has any traits apart from the usual ones

    all our components have at least 'config' and 'parent' as traitlets
    this is inherited from `traitlets.config.Configurable` so we ignore them
    here.
    '''
    return bool(
        set(cls.class_trait_names()) - set(ignore)
    )
