"""
Classes and functions related to telescope Optics
"""

import logging
from abc import abstractmethod
from enum import Enum, StrEnum, auto, unique

import astropy.units as u
import numpy as np
from astropy.table import QTable
from scipy.stats import laplace, laplace_asymmetric
from traitlets import validate

from ..core import TelescopeComponent
from ..core.traits import AstroQuantity, List, TraitError
from ..utils import get_table_dataset
from ..utils.quantities import all_to_value
from .warnings import warn_from_name

logger = logging.getLogger(__name__)

__all__ = [
    "OpticsDescription",
    "FocalLengthKind",
    "PSFModel",
    "ComaPSFModel",
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


class PSFModel(TelescopeComponent):
    """
    Base component to describe image distortion due to the optics of the different cameras.
    """

    @u.quantity_input(
        lon=u.deg,
        lat=u.deg,
        lon0=u.deg,
        lat0=u.deg,
    )
    @abstractmethod
    def pdf(self, lon, lat, lon0, lat0) -> np.ndarray:
        """
        Calculates the value of the psf at a given location.

        Parameters
        ----------
        lon : u.Quantity[angle]
            longitude coordinate of the point on the focal plane where the psf is evaluated
        lat : u.Quantity[angle]
            latitude coordinate of the point on the focal plane where the psf is evaluated
        lon0 : u.Quantity[angle]
            longitude coordinate of the point source on the focal plane
        lat0 : u.Quantity[angle]
            latitude coordinate of the point source on the focal plane
        Returns
        ----------
        psf : np.ndarray
            value of the PSF at the specified location with the specified position of the point source
        """

        pass


class ComaPSFModel(PSFModel):
    r"""
    PSF model describing pure coma aberrations PSF effect.

    The PSF is described by a product of an asymmetric Laplacian for the radial part and a symmetric Laplacian in the polar direction.
    Explicitly, the radial part is given by

    .. math:: f_{R}(r, K) = \begin{cases}\frac{1}{S_{R}(K+K^{-1})}e^{-K\frac{r-r_0}{S_{R}}}, r\ge r_0\\ \frac{1}{S_{R}(K+K^{-1})}e^{\frac{r-r_0}{KS_{R}}}, r < r_0\end{cases}

    and the polar part is given by

    .. math:: f_{\Phi}(\phi) = \frac{1}{2S_\phi}e^{-|\frac{\phi-\phi_0}{S_\phi}|}

    The parameters :math:`K`, :math:`S_{R}`, and :math:`S_{\phi}` are functions of the distance :math:`r` to the optical axis.
    Their detailed description is provided in the attributes section.

    Attributes
    ----------
    asymmetry_params : list
        Describes the dependency of the PSF on the distance to the optical axis.
        Used to calculate a PDF asymmetry parameter K of the asymmetric radial Laplacian
        of the PSF as a function of the distance r to the optical axis.

        .. math:: K(r) = 1 - c_0 \tanh(c_1 r) - c_2 r

    radial_scale_params : list
        Describes the dependency of the radial scale on the distance to the optical axis.
        Used to calculate width Sr of the asymmetric radial Laplacian in the PSF as a function of the distance :math:`r` to the optical axis.

        .. math:: S_{R}(r) = b_1 + b_2\,r + b_3\,r^2 + b_4\,r^3

    phi_scale_params : list
        Describes the dependency of the polar angle (:math:`\phi`) scale on the distance to the optical axis.
        Used to calculate the width Sf of the polar Laplacian in the PSF as a function of the distance :math:`r` to the optical axis.

        .. math:: S_{\phi}(r) = a_1\,\exp{(-a_2\,r)}+\frac{a_3}{a_3+r}

    Parameters
    ----------
    subarray : ctapipe.instrument.SubarrayDescription
        Description of the subarray.

    References
    ----------
    For reference, see :cite:p:`startracker`
    """

    asymmetry_params = List(
        help=(
            "Describes the dependency of the PSF on the distance "
            "to the optical axis. Used to calculate a PDF "
            "asymmetry parameter :math:`K` of the asymmetric radial Laplacian "
            "of the PSF as a function of the distance r to the optical axis"
        )
    ).tag(config=True)

    radial_scale_params = List(
        help=(
            "Describes the dependency of the radial scale on the "
            "distance to the optical axis. Used to calculate "
            "width :math:`S_R` of the asymmetric radial Laplacian in the PSF "
            "as a function of the distance r to the optical axis"
        )
    ).tag(config=True)

    phi_scale_params = List(
        help=(
            "Describes the dependency of the polar scale on the "
            "distance to the optical axis. Used to calculate "
            r"the width :math:`S_\phi` of the polar Laplacian in the PSF "
            "as a function of the distance r to the optical axis"
        )
    ).tag(config=True)

    pixel_width = AstroQuantity(
        default_value=0.1 * u.deg,
        physical_type=u.physical.angle,
        help="Width of a pixel in FoV coordinates",
    ).tag(config=True) 

    def _k(self, r):
        c1, c2, c3 = self.asymmetry_params
        return 1 - c1 * np.tanh(c2 * r) - c3 * r

    def _s_r(self, r):
        return np.polyval(self.radial_scale_params[::-1], r)

    def _s_phi(self, r):
        a1, a2, a3 = self.phi_scale_params
        return a1 * np.exp(-a2 * r) + a3 / (a3 + r)

    @u.quantity_input(
        lon=u.deg,
        lat=u.deg,
        lon0=u.deg,
        lat0=u.deg,
    )
    def pdf(self, lon, lat, lon0, lat0):
        lon, lat, lon0, lat0, pixel_width = all_to_value(
            lon, lat, lon0, lat0, self.pixel_width, unit=u.rad
        )

        r0 = np.sqrt(lon0**2 + lat0**2)

        # Evaluate PSF parameters at source position
        k = self._k(r0)
        s_r = self._s_r(r0)
        s_phi = self._s_phi(r0)

        dlon = lon - lon0
        dlat = lat - lat0

        r = np.sqrt(dlon**2 + dlat**2)
        phi = np.arctan2(dlat, dlon)

        radial_pdf = laplace_asymmetric.pdf(r, k, 0.0, s_r)
        polar_pdf = laplace.pdf(phi, 0.0, s_phi)

        at_center = np.isclose(r0, 0, atol=pixel_width)
        polar_pdf = np.where(at_center, 1 / (2 * s_phi), polar_pdf)

        # Polar PDF is valid under approximation that the polar axis is orthogonal to the radial axis
        # Thus, we limit the PDF to a chord of 6 pixels or covering ~30deg around the radial axis, whichever is smaller
        chord_length = min(6 * pixel_width, 0.5 * r0)

        if not np.isclose(r0, 0, atol=pixel_width):
            dphi = np.arcsin(chord_length / (2 * r0))
            mask = np.abs(phi) <= dphi
            polar_pdf = np.where(mask, polar_pdf, 0.0)

        return radial_pdf * polar_pdf

    @validate("asymmetry_params")
    def _check_asymmetry_params(self, proposal):
        if len(proposal["value"]) != 3:
            raise TraitError("asymmetry_params needs to have length 3")
        return proposal["value"]

    @validate("radial_scale_params")
    def _check_radial_scale_params(self, proposal):
        if len(proposal["value"]) != 4:
            raise TraitError("radial_scale_params needs to have length 4")
        return proposal["value"]

    @validate("phi_scale_params")
    def _check_phi_scale_params(self, proposal):
        if len(proposal["value"]) != 3:
            raise TraitError("phi_scale_params needs to have length 3")
        return proposal["value"]
