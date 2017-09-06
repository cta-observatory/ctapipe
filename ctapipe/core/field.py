import numpy as np
from astropy.units.core import UnitBase


class Field:
    '''
    Class for storing an arbitrary python object in a `Container`.

    Parameters
    ----------
    description: str
        Help text associated with the item
    default:
        default value of the item (this will be set when the `Container`
        is constructed, as well as when  `Container.reset()` is called
    allow_none: bool
        If true, the field is allowed to contain `None`
    ucd: str
        universal content descriptor (see Virtual Observatory standards)
    '''

    def __init__(
        self,
        description,
        default=None,
        allow_none=False,
        ucd=None
    ):
        self.allow_none = allow_none
        self.default = default
        self.description = description
        self.ucd = ucd

    def __repr__(self):
        desc = '{}'.format(self.description)
        return desc

    def coerce(self, value):
        if value is None and not self.allow_none:
            raise ValueError('Value must not be None')

        return value


class ArrayField(Field):
    '''
    Class for storing a numpy array in a `Container`.

    Parameters
    ----------
    description: str
        Help text associated with the item
    default:
        default value of the item (this will be set when the `Container`
        is constructed, as well as when  `Container.reset()` is called
    shape: tuple
        The shape of the field
    dtype: numpy.dtype or None
        If given, coerce values to this dtype
    allow_none: bool
        If true, the field is allowed to contain `None`
    ucd: str
        universal content descriptor (see Virtual Observatory standards)
    '''

    def __init__(
        self,
        description,
        default=None,
        shape=None,
        dtype=None,
        allow_none=False,
        ucd=None
    ):
        super().__init__(
            description=description,
            default=default,
            allow_none=allow_none,
            ucd=ucd,
        )
        self.shape = shape
        self.dtype = dtype

    def coerce(self, value):

        if value is None:
            if self.allow_none:
                return value
            else:
                raise ValueError('Value must not be None')

        value = np.asanyarray(value)

        if self.shape is not None and self.shape != value.shape:
            raise ValueError('New Value has wrong shape')

        if self.dtype is not None:
            value = value.astype(self.dtype)

        return value


class QuantityField(ArrayField):
    '''
    Class for storing an astropy quantity in a `Container`.

    Parameters
    ----------
    description: str
        Help text associated with the item
    unit: astropy.units.Unit
        The unit of the Field
    default:
        default value of the item (this will be set when the `Container`
        is constructed, as well as when  `Container.reset()` is called
    shape: tuple
        The shape of the field
    dtype: numpy.dtype or None
        If given, coerce values to this dtype
    allow_none: bool
        If true, the field is allowed to contain `None`
    ucd: str
        universal content descriptor (see Virtual Observatory standards)
    '''

    def __init__(
        self,
        description,
        unit,
        default=None,
        shape=None,
        dtype=None,
        allow_none=False,
        ucd=None
    ):
        super().__init__(
            description=description,
            default=default,
            allow_none=allow_none,
            ucd=ucd,
            shape=shape,
            dtype=dtype,
        )
        if not isinstance(unit, UnitBase):
            raise ValueError('Unit must be an astropy unit')
        self.unit = unit

    def coerce(self, value):

        value = super().coerce(value)

        if not self.unit.is_equivalent(value.unit):
            raise ValueError('New value has wrong unit')

        return value

    def __repr__(self):
        desc = '{}'.format(self.description)
        if self.unit is not None:
            desc += ' [{}]'.format(self.unit)
        return desc
