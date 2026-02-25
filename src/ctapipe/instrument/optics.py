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

from ..coordinates import TelescopeFrame
from ..core import TelescopeComponent
from ..core.traits import FloatTelescopeParameter
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
    def pdf(self, tel_id, lon, lat, lon0, lat0) -> np.ndarray:
        """
        Calculates the value of the psf at a given location.

        Parameters
        ----------
        tel_id : int
            ID of the telescope for which the PSF is evaluated
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


def _cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi


class ComaPSFModel(PSFModel):
    r"""
    PSF model describing pure coma aberrations PSF effect.

    The PSF is described by a product of an asymmetric Laplacian for the radial part and a symmetric Laplacian in the polar direction.
    Explicitly, the radial part is given by

    .. math:: f_{R}(r, K) = \begin{cases}\frac{1}{S_{R}(K+K^{-1})}e^{-K\frac{r-r_0}{S_{R}}}, r\ge r_0\\ \frac{1}{S_{R}(K+K^{-1})}e^{\frac{r-r_0}{KS_{R}}}, r < r_0\end{cases}

    and the polar part is given by

    .. math:: f_{\Phi}(\phi) = \frac{1}{2S_\phi}e^{-|\frac{\phi-\phi_0}{S_\phi}|}

    The parameters :math:`K`, :math:`S_{R}`, and :math:`S_{\phi}` are functions of the distance :math:`r` to the optical axis,
    configured via telescope traitlets:

    - Asymmetry parameters (:math:`K`):

        .. math:: K(r) = 1 - c_0 \tanh(c_1 r) - c_2 r

    - Radial scale parameters (:math:`S_R`):

        .. math:: S_{R}(r) = b_1 + b_2\,r + b_3\,r^2 + b_4\,r^3

    - Polar scale parameters (:math:`S_\phi`):

        .. math:: S_{\phi}(r) = a_1\,\exp{(-a_2\,r)}+\frac{a_3}{a_3+r}

    Parameters
    ----------
    subarray : ctapipe.instrument.SubarrayDescription
        Description of the subarray.

    References
    ----------
    For reference, see :cite:p:`startracker`
    """

    asymmetry_max = FloatTelescopeParameter(
        default_value=None,
        allow_none=True,
        help=r"Maximum asymmetry parameter K at large distance from the optical axis (:math:`c_0`)",
    ).tag(config=True)

    asymmetry_decay_rate = FloatTelescopeParameter(
        default_value=None,
        allow_none=True,
        help=r"Tanh saturation rate of the asymmetry parameter K with distance to the optical axis (:math:`c_1`)",
    ).tag(config=True)

    asymmetry_linear_term = FloatTelescopeParameter(
        default_value=None,
        allow_none=True,
        help=r"Linear term for the asymmetry parameter K with distance to the optical axis (:math:`c_2`)",
    ).tag(config=True)

    radial_scale_offset = FloatTelescopeParameter(
        default_value=None,
        allow_none=True,
        help=r"Offset of the radial scale :math:`S_R` (:math:`b_1`)",
    ).tag(config=True)

    radial_scale_linear = FloatTelescopeParameter(
        default_value=None,
        allow_none=True,
        help=r"Linear growth of the radial scale :math:`S_R` with distance to the optical axis (:math:`b_2`)",
    ).tag(config=True)

    radial_scale_quadratic = FloatTelescopeParameter(
        default_value=None,
        allow_none=True,
        help=r"Quadratic growth of the radial scale :math:`S_R` with distance to the optical axis (:math:`b_3`)",
    ).tag(config=True)

    radial_scale_cubic = FloatTelescopeParameter(
        default_value=None,
        allow_none=True,
        help=r"Cubic growth of the radial scale :math:`S_R` with distance to the optical axis (:math:`b_4`)",
    ).tag(config=True)

    polar_scale_amplitude = FloatTelescopeParameter(
        default_value=None,
        allow_none=True,
        help=r"Initial width :math:`S_\phi` at the center of the camera (r=0) (:math:`a_1`)",
    ).tag(config=True)

    polar_scale_decay = FloatTelescopeParameter(
        default_value=None,
        allow_none=True,
        help=r"Exponential decay of the polar scale :math:`S_\phi` with distance to the optical axis (:math:`a_2`)",
    ).tag(config=True)

    polar_scale_offset = FloatTelescopeParameter(
        default_value=None,
        allow_none=True,
        help=r"Offset controlling width :math:`S_\phi` at large distance from the optical axis (:math:`a_3`)",
    ).tag(config=True)

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        """Initialize the ComaPSFModel component and check for missing configuration parameters."""

        super().__init__(
            subarray=subarray,
            config=config,
            parent=parent,
            **kwargs,
        )
        # Get the pixel size in degrees for the given telescopes.
        self.pixel_width = {}
        for tel_id in self.subarray.tels:
            cam_geom = self.subarray.tel[tel_id].camera.geometry
            # Transform camera geometry to telescope frame if not already in that frame
            if cam_geom.frame != TelescopeFrame:
                cam_geom = cam_geom.transform_to(TelescopeFrame())

            # pixel width is only used to determine a useful distance measure to the camera center
            self.pixel_width[tel_id] = cam_geom.pixel_width[0].to_value(u.deg)

        # Check for missing config parameters and raise an error if any are missing.
        missing_config_parameters = []
        config_parameter_names = [
            name
            for name, trait in self.class_traits().items()
            if isinstance(trait, FloatTelescopeParameter)
        ]
        for name in config_parameter_names:
            config_parameter = getattr(self, name)
            if config_parameter.tel[self.subarray.tel_ids[0]] is None:
                missing_config_parameters.append(name)

        if missing_config_parameters:
            raise ValueError(
                f"Missing ComaPSFModel configuration parameters: {missing_config_parameters}"
            )

    def _k(self, tel_id, r):
        c0 = self.asymmetry_max.tel[tel_id]
        c1 = self.asymmetry_decay_rate.tel[tel_id]
        c2 = self.asymmetry_linear_term.tel[tel_id]
        return 1 - c0 * np.tanh(c1 * r) - c2 * r

    def _s_r(self, tel_id, r):
        return np.polyval(
            [
                self.radial_scale_cubic.tel[tel_id],
                self.radial_scale_quadratic.tel[tel_id],
                self.radial_scale_linear.tel[tel_id],
                self.radial_scale_offset.tel[tel_id],
            ],
            r,
        )

    def _s_phi(self, tel_id, r):
        a1 = self.polar_scale_amplitude.tel[tel_id]
        a2 = self.polar_scale_decay.tel[tel_id]
        a3 = self.polar_scale_offset.tel[tel_id]
        return a1 * np.exp(-a2 * r) + a3 / (a3 + r)

    @u.quantity_input(
        lon=u.deg,
        lat=u.deg,
        lon0=u.deg,
        lat0=u.deg,
    )
    def pdf(self, tel_id, lon, lat, lon0, lat0):
        # Convert all inputs to degrees for the calculations
        lon, lat, lon0, lat0 = all_to_value(lon, lat, lon0, lat0, unit=u.deg)

        r, phi = _cartesian_to_polar(lon, lat)
        r0, phi0 = _cartesian_to_polar(lon0, lat0)

        # Evaluate PSF parameters at source position
        k = self._k(tel_id, r0)
        s_r = self._s_r(tel_id, r0)
        s_phi = self._s_phi(tel_id, r0)

        radial_pdf = laplace_asymmetric.pdf(r, k, r0, s_r)
        polar_pdf = laplace.pdf(phi, phi0, s_phi)

        at_center = np.isclose(r0, 0, atol=self.pixel_width[tel_id])
        polar_pdf = np.where(at_center, 1 / (2 * s_phi), polar_pdf)

        # Polar PDF is valid under approximation that the polar axis is orthogonal to the radial axis
        # Thus, we limit the PDF to a chord of 6 pixels or covering ~30deg around the radial axis, whichever is smaller
        chord_length = min(6 * self.pixel_width[tel_id], 0.5 * r0)

        if r0 != 0:
            dphi = np.arcsin(chord_length / (2 * r0))
            polar_pdf[phi < phi0 - dphi] = 0
            polar_pdf[phi > phi0 + dphi] = 0

        return radial_pdf * polar_pdf
