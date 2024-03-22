"""
Classes and functions related to telescope Optics
"""

import logging
from enum import Enum, auto, unique

import astropy.units as u
import numpy as np
from astropy.table import QTable

from ..compat import StrEnum
from ..utils import get_table_dataset
from .warnings import warn_from_name

logger = logging.getLogger(__name__)

__all__ = [
    "OpticsDescription",
    "FocalLengthKind",
]


@unique
class FocalLengthKind(Enum):
    """
    Enumeration for the different kinds of focal lengths.
    """

    #: Effective focal length computed from ray tracing a point source
    #: and calculating the off-axis center of mass of the light distribution.
    #: This focal length should be used in coordinate transforms between camera
    #: frame and telescope frame to correct for the mean effect of coma aberration.
    EFFECTIVE = auto()
    #: Equivalent focal length is the nominal focal length of the main reflector
    #: for single mirror telescopes and the thin-lens equivalent for dual mirror
    #: telescopes.
    EQUIVALENT = auto()


@unique
class SizeType(StrEnum):
    """
    Enumeration of different telescope sizes (LST, MST, SST)
    """

    #: Unknown
    UNKNOWN = "UNKNOWN"
    #: A telescope with a mirror diameter larger than 16m
    LST = "LST"
    #: A telescope with a mirror diameter larger than 8m
    MST = "MST"
    #: Telescopes with a mirror diameter smaller than 8m
    SST = "SST"


@unique
class ReflectorShape(Enum):
    """
    Enumeration of the different reflector shapes
    """

    #: Unknown
    UNKNOWN = "UNKNOWN"
    #: A telescope with a parabolic dish
    PARABOLIC = "PARABOLIC"
    #: A telescope with a Davies--Cotton dish
    DAVIES_COTTON = "DAVIES_COTTON"
    #: A telescope with a hybrid between parabolic and Davies--Cotton dish
    HYBRID = "HYBRID"
    #: A dual mirror Schwarzschild-Couder reflector
    SCHWARZSCHILD_COUDER = "SCHWARZSCHILD_COUDER"


class OpticsDescription:
    """
    Describes the optics of a Cherenkov Telescope mirror

    The string representation of an `OpticsDescription` will be a combination
    of the telescope-type and sub-type as follows: "type-subtype". You can
    also get each individually.

    Parameters
    ----------
    name : str
        Name of this optical system
    n_mirrors : int
        Number of mirrors, i. e. 2 for Schwarzschild-Couder else 1
    equivalent_focal_length : astropy.units.Quantity[length]
        Equivalent focal-length of telescope, independent of which type of
        optics (as in the Monte-Carlo). This is the nominal focal length
        for single mirror telescopes and the equivalent focal length for dual
        mirror telescopes.
    effective_focal_length : astropy.units.Quantity[length]
        The effective_focal_length is the focal length estimated from
        ray tracing to correct for coma aberration. It is thus not automatically
        available for all simulations, but only if it was set beforehand
        in the simtel configuration. This is the focal length that should be
        used for transforming from camera frame to telescope frame for all
        reconstruction tasks to correct for the mean aberration.
    mirror_area : astropy.units.Quantity[area]
        total reflective surface area of the optical system (in m^2)
    n_mirror_tiles : int
        number of mirror facets

    Raises
    ------
    ValueError:
        if tel_type or mirror_type are not one of the accepted values
    TypeError, astropy.units.UnitsError:
        if the units of one of the inputs are missing or incompatible
    """

    CURRENT_TAB_VERSION = "4.0"
    COMPATIBLE_VERSIONS = {"4.0"}

    __slots__ = (
        "name",
        "size_type",
        "effective_focal_length",
        "equivalent_focal_length",
        "mirror_area",
        "n_mirrors",
        "n_mirror_tiles",
        "reflector_shape",
    )

    @u.quantity_input(
        mirror_area=u.m**2,
        equivalent_focal_length=u.m,
        effective_focal_length=u.m,
    )
    def __init__(
        self,
        name,
        size_type,
        n_mirrors,
        equivalent_focal_length,
        effective_focal_length,
        mirror_area,
        n_mirror_tiles,
        reflector_shape,
    ):
        self.name = name
        self.size_type = SizeType(size_type)
        self.reflector_shape = ReflectorShape(reflector_shape)
        self.equivalent_focal_length = equivalent_focal_length.to(u.m)
        self.effective_focal_length = effective_focal_length.to(u.m)
        self.mirror_area = mirror_area
        self.n_mirrors = n_mirrors
        self.n_mirror_tiles = n_mirror_tiles

    def __hash__(self):
        """Make this hashable, so it can be used as dict keys or in sets"""
        # From python >= 3.10, hash of nan is random, we want a fixed hash also for
        # unknown effective focal length:
        if np.isnan(self.effective_focal_length.value):
            effective_focal_length = -1
        else:
            effective_focal_length = self.effective_focal_length.to_value(u.m)

        return hash(
            (
                round(self.equivalent_focal_length.to_value(u.m), 4),
                round(effective_focal_length, 4),
                round(self.mirror_area.to_value(u.m**2)),
                self.size_type.value,
                self.reflector_shape.value,
                self.n_mirrors,
                self.n_mirror_tiles,
            )
        )

    def __eq__(self, other):
        """For eq, we just compare equal hash"""
        return hash(self) == hash(other)

    @classmethod
    def from_name(cls, name, optics_table="optics"):
        """
        Construct an OpticsDescription from the name.

        This needs the ``optics`` table dataset to be accessible via
        ``~ctapipe.utils.get_table_dataset``.

        Parameters
        ----------
        name: str
            string representation of optics (MST, LST, SST-1M, SST-ASTRI,...)
        optics_table: str
            base filename of optics table if not 'optics.*'


        Returns
        -------
        OpticsDescription

        """
        warn_from_name()

        if isinstance(optics_table, str):
            table = get_table_dataset(optics_table, role="OpticsDescription.from_name")
        else:
            table = optics_table

        version = table.meta.get("TAB_VER")

        if version not in cls.COMPATIBLE_VERSIONS:
            raise ValueError(f"Unsupported version of optics table: {version}")

        mask = table["optics_name"] == name

        (idx,) = np.nonzero(mask)
        if len(idx) == 0:
            raise ValueError(f"Unknown optics name {name}")

        # QTable so that accessing row[col] is a quantity
        table = QTable(table)
        row = table[idx[0]]

        return cls(
            name=name,
            size_type=row["size_type"],
            reflector_shape=row["reflector_shape"],
            n_mirrors=row["n_mirrors"],
            equivalent_focal_length=row["equivalent_focal_length"],
            effective_focal_length=row["effective_focal_length"],
            mirror_area=row["mirror_area"],
            n_mirror_tiles=row["n_mirror_tiles"],
        )

    @classmethod
    def get_known_optics_names(cls, optics_table="optics"):
        """
        return the list of optics names from ctapipe resources, i.e. those that can be
        constructed by name (this does not return the list of known names from an
        already open Monte-Carlo file)

        Parameters
        ----------
        optics_table: str or astropy Table
            table where to read the optics description from. If a string, this is
            opened with `ctapipe.utils.get_table_dataset()`
        """
        if isinstance(optics_table, str):
            table = get_table_dataset(optics_table, role="get_known_optics_names")
        else:
            table = optics_table
        return np.array(table["name"])

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}"
            f", size_type={self.size_type.value}"
            f", reflector_shape={self.reflector_shape.value}"
            f", equivalent_focal_length={self.equivalent_focal_length:.2f}"
            f", effective_focal_length={self.effective_focal_length:.2f}"
            f", n_mirrors={self.n_mirrors}"
            f", mirror_area={self.mirror_area:.2f}"
            ")"
        )

    def __str__(self):
        return self.name
