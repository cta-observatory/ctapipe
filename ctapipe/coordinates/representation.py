"""
This module defines any reference systems which may be needed in addition
"""
from astropy.coordinates import (
    BaseRepresentation,
    CartesianRepresentation,
)
import astropy.units as u
from collections import OrderedDict
from numpy import broadcast_arrays


class PlanarRepresentation(BaseRepresentation):
    """
    Representation of a point in a 2D plane.
    This is essentially a copy of the Cartesian representation used
    in astropy.

    Parameters
    ----------

    x, y : `~astropy.units.Quantity`
        The x and y coordinates of the point(s). If ``x`` and ``y``have
        different shapes, they should be broadcastable.

    copy : bool, optional
        If True arrays will be copied rather than referenced.

    """
    attr_classes = OrderedDict([('x', u.Quantity),
                                ('y', u.Quantity)])

    def __init__(self, x, y, copy=True, **kwargs):

        if x is None or y is None:
            raise ValueError(
                'x and y are required to instantiate CartesianRepresentation'
            )

        if not isinstance(x, self.attr_classes['x']):
            raise TypeError('x should be a {0}'.format(self.attr_classes['x'].__name__))

        if not isinstance(y, self.attr_classes['y']):
            raise TypeError('y should be a {0}'.format(self.attr_classes['y'].__name__))

        x = self.attr_classes['x'](x, copy=copy)
        y = self.attr_classes['y'](y, copy=copy)

        if not (x.unit.physical_type == y.unit.physical_type):
            raise u.UnitsError('x and y should have matching physical types')

        try:
            x, y = broadcast_arrays(x, y, subok=True)
        except ValueError:
            raise ValueError('Input parameters x and y cannot be broadcast')

        self._x = x
        self._y = y
        self._differentials = {}

    @property
    def x(self):
        """
        The x component of the point(s).
        """
        return self._x

    @property
    def y(self):
        """
        The y component of the point(s).
        """
        return self._y

    @property
    def xy(self):
        return u.Quantity((self._x, self._y))

    @classmethod
    def from_cartesian(cls, cartesian):

        return cls(x=cartesian.x, y=cartesian.y)

    def to_cartesian(self):
        return CartesianRepresentation(
            x=self._x, y=self._y, z=0 * self._x.unit
        )

    @property
    def components(self):
        return 'x', 'y'
