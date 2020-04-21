import warnings
from abc import ABCMeta, abstractmethod
import astropy.units as u
import numpy as np


class Field(metaclass=ABCMeta):
    """
    Abstract base class for defining data storage in `Container` classes.

    Attributes
    ----------
    default:
        default value of the field
    description: str
        Help text associated with the field
    allow_none: bool
        If ``None`` is an allowed value for this field or not
    """

    def __init__(self, default, description, allow_none=False):
        self.default = default
        self.description = description
        self.allow_none = allow_none

    @abstractmethod
    def validate(self, value):
        '''
        Validate if ```value``` conforms with the requirements of this field.
        Validation should also transforms into the nominal representation of the field.
        '''
        if value is None and not self.allow_none:
            raise ValueError('must not be None')
        return value

    def __repr__(self):
        return f"{self.__class__name}(default={self.default})"

    def __str__(self):
        return self.__repr__() + ": {self.description}"


class TypedField(Field):
    '''ABC for fields consisting of a simple python type'''

    @property
    @classmethod
    @abstractmethod
    def type_(cls):
        '''Python type of the Field, can be set as a simple class variable'''
        pass

    def validate(self, value):
        value = super().validate(value)
        if value is None:
            return None

        return self.type_(value)


class FloatField(TypedField):
    '''A field consisting of a single python `float`

    Attributes
    ----------
    default: float or None
        default value of the field
    description: str
        Help text associated with the field
    allow_none: bool
        If ``None`` is an allowed value for this field or not
    '''
    type_ = float


class IntField(TypedField):
    '''A field consisting of a single python `int`

    Attributes
    ----------
    default: int
        default value of the field
    description: str
        Help text associated with the item
    allow_none: bool
        If ``None`` is an allowed value for this field or not
    '''
    type_ = int


class BoolField(TypedField):
    '''A field consisting of a single python `bool`

    Attributes
    ----------
    default: bool
        default value of the field
    description: str
        Help text associated with the item
    allow_none: bool
        If ``None`` is an allowed value for this field or not
    '''
    type_int = bool


class StringField(TypedField):
    '''
    A field consisting of a python str.
    Max length needs to be specified, because most data formats
    do not support arbitrary length strings.
    Strings longer than max_length will be truncated.

    default: str
        default value of the field
    description: str
        Help text associated with the item
    allow_none: bool
        If ``None`` is an allowed value for this field or not
    '''
    type_ = str

    def __init__(self, default, description, max_length):
        super().__init__(default=default, description=description)
        self.max_length = max_length

    def validate(self, value):
        super().validate()
        if value is None:
            return None

        if not isinstance(value, str):
            raise TypeError('must be string')

        if len(value) <= self.max_length:
            return value

        warnings.warn(f'String {value!r} is longer than {self.max_length} characters, truncating')
        return value[:self.max_length]


class QuantityField(Field):
    '''
    A field consisting of an astropy quantity.
    This field should also be used for generic arrays, the default
    unit is dimensionless_unscaled.

    default: str
        default value of the field
    description: str
        Help text associated with the item
    allow_none: bool
        If ``None`` is an allowed value for this field or not
    unit: astropy.units.Unit
        unit of the field.
        Convertible units will be converted, others will raise errors.
    ndim: int
        Dimensionality of the quantity, 0 is a single number
    shape: tuple[int] or None
        if given, a fixed shape is required
    '''
    def __init__(
        self, *args,
        unit=u.dimensionless_unscaled,
        dtype=np.float64,
        ndim=0,
        shape=None,
        **kwargs
    ):
        super().__init__(self, *args, **kwargs)
        self.unit = u.Unit(unit)
        self.ndim = ndim
        self.shape = None if shape is None else tuple(shape)
        self.dtype = np.dtype(dtype)

    def validate(self, value):
        value = super().validate(value)

        value = np.asanyarray(value)

        if value.ndim != self.ndim:
            raise ValueError(f'wrong dimensionality {value.ndim}')

        if self.shape is not None and self.shape != value.shape:
            raise ValueError(f'wrong shape {value.shape}')

        value = value.astype(self.dtype, casting='safe')
        value = u.Quantity(value, dtype=self.dtype, copy=False, unit=self.unit)

        return value

    def __str__(self):
        return (
            f'QuantityField(unit={self.unit}, ndim={self.ndim}'
            f', shape={self.shape}, dtype={self.dtype}'
            ')'
        )
