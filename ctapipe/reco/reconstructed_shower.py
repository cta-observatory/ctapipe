# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simple reconstructed shower parameters container

At the moment this provides raw values for alt and az, but I think ultimately the
value of interest should be a coordinate object of the shower direction (in the direction
variable), this would allow the user to easily move between systems.

TODO:
-----

- Is it possible to force different objects to only accept a give variable type?
e.g. direction variable only takes AltAz object (or nominal system)

- This may want to be wrapped within a class to add further functionality

"""
from collections import namedtuple

__all__ = [
    'ReconstructedShower'
]

ReconstructedShower = namedtuple(
    "ReconstructedShower",
    "alt,az,direction,direction_uncertainty,"
    "core_x,core_y,core,core_uncertainty,"
    "energy,energy_uncertainty,"
    "hmax,hmax_uncertainty,"
    "xmax,xmax_uncertainty"
)

