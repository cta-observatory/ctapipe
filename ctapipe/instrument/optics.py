"""
Classes and functions related to telescope Optics
"""

import logging

import astropy.units as u
import numpy as np

from ..utils import get_table_dataset

logger = logging.getLogger(__name__)


class OpticsDescription:
    """
    Describes the optics of a Cherenkov Telescope mirror

    The string representation of an `OpticsDescription` will be a combination
    of the telescope-type and sub-type as follows: "type-subtype". You can
    also get each individually.

    Parameters
    ----------
    num_mirrors: int
        Number of mirrors, i. e. 2 for Schwarzschild-Couder else 1
    equivalent_focal_length : astropy.units.Quantity[length]
        Equivalent focal-length of telescope, independent of which type of
        optics (as in the Monte-Carlo). This is the nominal focal length
        for single mirror telescopes and the equivalent focal length for dual
        mirror telescopes.
    effective_focal_length : astropy.units.Quantity[length]
        The effective_focal_length is the focal length estimated from
        ray tracing to correct for koma aberration. It is thus not automatically
        available for all simulations, but only if it was set before hand
        in the simtel configuration. This is the focal length that should be
        used for transforming from camera frame to telescope frame for all
        reconstruction tasks to correct for the mean aberration.
    mirror_area: astropy.units.Quantity[area]
        total reflective surface area of the optical system (in m^2)
    num_mirror_tiles: int
        number of mirror facets

    Raises
    ------
    ValueError:
        if tel_type or mirror_type are not one of the accepted values
    TypeError, astropy.units.UnitsError:
        if the units of one of the inputs are missing or incompatible
    """

    __slots__ = (
        "name",
        "effective_focal_length",
        "equivalent_focal_length",
        "mirror_area",
        "num_mirrors",
        "num_mirror_tiles",
    )

    @u.quantity_input(
        mirror_area=u.m**2,
        equivalent_focal_length=u.m,
        effective_focal_length=u.m,
    )
    def __init__(
        self,
        name,
        num_mirrors,
        equivalent_focal_length,
        effective_focal_length,
        mirror_area,
        num_mirror_tiles,
    ):

        self.name = name
        self.equivalent_focal_length = equivalent_focal_length.to(u.m)
        self.effective_focal_length = effective_focal_length.to(u.m)
        self.mirror_area = mirror_area
        self.num_mirrors = num_mirrors
        self.num_mirror_tiles = num_mirror_tiles

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
                self.num_mirrors,
                self.num_mirror_tiles,
            )
        )

    def __eq__(self, other):
        """Make this hashable, so it can be used as dict keys or in sets"""
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
        if isinstance(optics_table, str):
            table = get_table_dataset(optics_table, role="OpticsDescription.from_name")
        else:
            table = optics_table

        version = table.meta.get("TAB_VER")

        # we introduced the TAB_VER after switching to the second version
        # of this table, so when the version is missing, it can be either 1 or 2
        # we guess the version by looking for the mirror_type attribute.
        if version is None:
            if "mirror_type" in table.colnames:
                version = "1.0"
            else:
                version = "2.0"

        if version not in {"1.0", "2.0"}:
            raise ValueError(f"Unsupported version of optics table: {version}")

        if version == "1.0":
            mask = table["tel_description"] == name
        else:
            mask = table["description"] == name

        if np.count_nonzero(mask) == 0:
            raise ValueError(f"Unknown telescope name {name}")

        if version == "1.0":
            num_mirrors = 1 if table["mirror_type"][mask][0] == "DC" else 2
        else:
            num_mirrors = table["num_mirrors"][mask][0]

        if version in {"1.0", "2.0"}:
            eff_focal_length = np.nan * u.m
        else:
            eff_focal_length = table["effective_focal_length"][mask].quantity[0]

        focal_length = table["equivalent_focal_length"][mask].quantity[0]

        return cls(
            name=name,
            num_mirrors=num_mirrors,
            equivalent_focal_length=focal_length,
            effective_focal_length=eff_focal_length,
            mirror_area=table["mirror_area"][mask].quantity[0],
            num_mirror_tiles=table["num_mirror_tiles"][mask][0],
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
        return np.array(table["tel_description"])

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(name={self.name}"
            f", equivalent_focal_length={self.equivalent_focal_length:.2f}"
            f", effective_focal_length={self.effective_focal_length:.2f}"
            f", num_mirrors={self.num_mirrors}"
            f", mirror_area={self.mirror_area:.2f}"
            ")"
        )

    def __str__(self):
        return self.name
