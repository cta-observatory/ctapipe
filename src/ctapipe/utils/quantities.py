"""utils to deal with astropy.units.Quantity"""

from astropy.units import Quantity


def all_to_value(*args, unit):
    """Convert quantities to numeric values in a common unit.

    Unit conversions raise an error for incompatible inputs. The result can
    share data with the inputs when conversion is unnecessary.

    Parameters
    ----------
    *args
        Quantities to convert, in the order in which the numeric values should be
        returned.
    unit : astropy.units.UnitBase or str
        Common unit in which to express the values.

    Returns
    -------
    tuple
        Numeric values in the requested unit, in input order.
    """
    return tuple(Quantity(arg, copy=False).to_value(unit) for arg in args)
