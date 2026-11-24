"""CTAPipe deprecation system."""

import astropy.utils

__all__ = [
    "CTAPipeDeprecationWarning",
    "deprecated",
    "deprecated_renamed_argument",
    "deprecated_attribute",
]


class CTAPipeDeprecationWarning(Warning):
    """The CTAPipe deprecation warning."""


def deprecated(since, **kwargs):
    """Use to mark a function or class as deprecated.

    Reuses Astropy's deprecated decorator.
    Check arguments and usage in `~astropy.utils.decorator.deprecated`

    Parameters
    ----------
    since : str
        The release at which this API became deprecated.  This is required.
    **kwargs
        Options forwarded to Astropy's deprecation decorator, such as
        ``message`` and ``alternative``. The warning type is always
        ``CTAPipeDeprecationWarning``.
    """
    kwargs["warning_type"] = CTAPipeDeprecationWarning
    return astropy.utils.deprecated(since, **kwargs)


def deprecated_renamed_argument(old_name, new_name, since, **kwargs):
    """Deprecate a _renamed_ or _removed_ function argument.

    Check arguments and usage in `~astropy.utils.decorator.deprecated_renamed_argument`

    Parameters
    ----------
    old_name : str
        Name of the deprecated function argument.
    new_name : str or None
        Replacement argument name. Use None when the argument is removed without a
        replacement.
    since : str
        Release in which the old argument became deprecated.
    **kwargs
        Options forwarded to Astropy's renamed-argument decorator. ctapipe selects the
        warning class.
    """
    kwargs["warning_type"] = CTAPipeDeprecationWarning
    return astropy.utils.deprecated_renamed_argument(
        old_name, new_name, since, **kwargs
    )


def deprecated_attribute(name, since, **kwargs):
    """Use to mark a public attribute as deprecated.

    This creates a
    property that will warn when the given attribute name is accessed.

    Parameters
    ----------
    name : str
        Name of the attribute being deprecated.
    since : str
        Release in which the attribute became deprecated.
    **kwargs
        Options forwarded to Astropy's attribute-deprecation helper. ctapipe selects the
        warning class.
    """
    kwargs["warning_type"] = CTAPipeDeprecationWarning
    return astropy.utils.deprecated_attribute(name, since, **kwargs)
