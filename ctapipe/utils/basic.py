from inspect import isabstract


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
