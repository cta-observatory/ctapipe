def _fix_user_options(options):
    """
    This is for Python 2.x and 3.x compatibility.  distutils expects Command
    options to all be byte strings on Python 2 and Unicode strings on Python 3.
    """

    def to_str_or_none(x):
        if x is None:
            return None
        return str(x)

    return [tuple(to_str_or_none(x) for x in y) for y in options]
