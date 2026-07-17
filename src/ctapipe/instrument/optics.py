"""
Classes and functions related to telescope Optics
"""

import logging
from abc import abstractmethod
from enum import Enum, StrEnum, auto, unique
from functools import cached_property
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import QTable, Table
from numpy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.stats import laplace, laplace_asymmetric
from zernike import RZern

from ..coordinates import TelescopeFrame
from ..core import TelescopeComponent
from ..core.traits import (
    AstroQuantity,
    Float,
    FloatTelescopeParameter,
    Int,
    TelescopeParameter,
)
from ..utils import get_table_dataset
from ..utils.quantities import all_to_value
from .warnings import warn_from_name

logger = logging.getLogger(__name__)

__all__ = [
    "OpticsDescription",
    "FocalLengthKind",
    "PSFModel",
    "ComaPSFModel",
    "ZernikePSFModel",
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

    #: Version of the legacy, per-subarray optics table
    CURRENT_TAB_VERSION = "4.0"
    COMPATIBLE_VERSIONS = {"4.0"}

    #: Version for the new, per-telescope table that will allow describing
    #: mirror facets in the future
    CURRENT_TEL_TAB_VERSION = "1.0"
    COMPATIBLE_TEL_TAB_VERSIONS = {"1.0"}

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

    def to_table(self):
        """
        Convert this OpticsDescription to an astropy Table for writing to files.

        See `OpticsDescription.from_table` for the opposite operation.
        """
        table = Table()
        table.meta["TAB_VER"] = self.CURRENT_TEL_TAB_VERSION
        table.meta["optics_name"] = self.name
        table.meta["size_type"] = self.size_type.value
        table.meta["equivalent_focal_length"] = self.equivalent_focal_length.to_value(
            u.m
        )
        table.meta["effective_focal_length"] = self.effective_focal_length.to_value(u.m)
        table.meta["reflector_shape"] = self.reflector_shape.value
        table.meta["n_mirrors"] = self.n_mirrors
        table.meta["n_mirror_tiles"] = self.n_mirror_tiles
        table.meta["mirror_area"] = self.mirror_area.to_value(u.m**2)
        return table

    @classmethod
    def from_table(cls, table: Table | str | Path, **kwargs):
        """
        Create an OpticsDescription instance from an astropy Table.

        See `OpticsDescription.to_table` for the opposite operation.
        """
        if not isinstance(table, Table):
            table = Table.read(table, **kwargs)

        version = table.meta.get("TAB_VER")
        if version not in cls.COMPATIBLE_TEL_TAB_VERSIONS:
            raise OSError(f"Unsupported telescope optics table version: {version}")

        return cls(
            name=table.meta["optics_name"],
            size_type=SizeType(table.meta["size_type"]),
            equivalent_focal_length=u.Quantity(
                table.meta["equivalent_focal_length"], u.m
            ),
            effective_focal_length=u.Quantity(
                table.meta["effective_focal_length"], u.m
            ),
            n_mirrors=table.meta["n_mirrors"],
            n_mirror_tiles=table.meta["n_mirror_tiles"],
            reflector_shape=ReflectorShape(table.meta["reflector_shape"]),
            mirror_area=u.Quantity(table.meta["mirror_area"], u.m**2),
        )

    def get_focal_length(self, focal_length_choice=FocalLengthKind.EFFECTIVE):
        """
        Get focal length for coordinate transformations.

        This is a helper function to get the focal length, mainly
        to attach it to the ``CameraFrame`` coordinate frame for
        pixel coordinate transformations.

        In most cases, the effective focal length should be strongly preferred,
        as it takes the effect of optical aberrations on the plate scale into account.

        By default, this function will try to use the effective focal length and raise
        an error if it is not available.
        """
        if isinstance(focal_length_choice, str):
            focal_length_choice = FocalLengthKind[focal_length_choice.upper()]

        if focal_length_choice is FocalLengthKind.EFFECTIVE:
            focal_length = self.effective_focal_length
            if np.isnan(focal_length.value):
                raise RuntimeError(
                    "`focal_length_choice` was set to 'EFFECTIVE', but the"
                    " effective focal length was not present in the input. "
                    " Set `focal_length_choice='EQUIVALENT'` or make sure"
                    " input files contain the effective focal length"
                )
            return focal_length

        if focal_length_choice is FocalLengthKind.EQUIVALENT:
            return self.equivalent_focal_length

        raise ValueError(f"Invalid focal length choice: {focal_length_choice}")


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

    Notes
    -----
    **Model Limitations:**

    - For sources within the central pixel (r0 < pixel_width), the PSF is approximated
      as a uniform distribution over the pixel area. This avoids singularities at the
      origin and provides a numerically stable approximation for sub-pixel sources.

    - The angular distribution is limited to a chord around the radial axis based on
      the approximation that the polar axis is orthogonal to the radial axis. This
      limits its validity to the inner camera region.

    - This PSF model is designed for pointing determination via star tracking. As the
      telescope tracks celestial coordinates, stars rotate around the camera center.
      Their reconstructed positions provide information about the telescope pointing
      direction. To achieve good pointing accuracy, stars should be located away from
      the camera center to provide a sufficient lever arm for the pointing solution.

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
        delta_phi = (phi - phi0 + np.pi) % (2 * np.pi) - np.pi
        polar_pdf = laplace.pdf(delta_phi, 0, s_phi)

        # Polar PDF is valid under approximation that the polar axis is orthogonal to the radial axis
        # Thus, we limit the PDF to a chord of 6 pixels or covering ~30deg around the radial axis, whichever is smaller
        chord_length = min(6 * self.pixel_width[tel_id], 0.5 * r0)

        pixel_radius = 0.5 * self.pixel_width[tel_id]

        if (
            r0 > pixel_radius
        ):  # only apply the chord limit for sources outside the central pixel
            dphi = np.arcsin(chord_length / (2 * r0))
            polar_pdf = np.where(np.abs(delta_phi) <= dphi, polar_pdf, 0.0)

        # If the source is within the central pixel, use uniform distribution inside pixel
        source_in_pixel = r0 < pixel_radius
        if source_in_pixel:
            # Uniform distribution inside the pixel, zero outside
            in_pixel = r < pixel_radius
            uniform_density = 1.0 / (np.pi * pixel_radius**2)
            pdf = np.where(in_pixel, uniform_density, 0.0)
        else:
            # Normal model: asymmetric Laplacian radial and Laplacian angular with 1/r Jacobian
            inv_r = np.divide(1.0, r, where=r != 0, out=np.zeros_like(r, dtype=float))
            pdf = radial_pdf * polar_pdf * inv_r

        return pdf


class ZernikePSFModel(PSFModel):
    r"""PSF model based on wavefront reconstruction using Zernike wavefront coefficients.

    This model reconstructs the optical wavefront from a set of Zernike
    polynomial coefficients and computes the point spread function (PSF)
    by Fourier propagation through the telescope pupil. The resulting PSF
    naturally includes diffraction and wavefront aberrations and is
    evaluated for arbitrary field positions by allowing selected Zernike
    coefficients to vary with the source position in the focal plane.

    The implementation uses:
    - Zernike polynomials in Noll indexing to describe the optical path
      difference (OPD) across the telescope pupil.
    - Scalar Fourier optics to propagate the complex pupil field into the
      focal plane.
    - Polychromatic averaging over the Cherenkov emission spectrum using a
      configurable wavelength range and spectral weighting.
    - Optional Gaussian smoothing to approximate detector and residual
      instrumental broadening not explicitly included in the wavefront
      model.

    In the current parameterization, field-dependent aberrations are
    represented by linear coma terms and quadratic astigmatism terms,
    providing a compact phenomenological description of off-axis optical
    degradation while retaining a physically motivated wavefront model.
    """

    # Universal model performance parameters
    pupil_size = Int(
        default_value=256,
        help=(
            "Number of samples across the FFT grid used to discretize the pupil. "
            "Larger values improve numerical accuracy and PSF sampling at the "
            "expense of increased memory usage and computation time."
        ),
    ).tag(config=True)

    pupil_diameter_fraction = Float(
        default_value=0.12,
        help=(
            "Diameter of the telescope pupil as a fraction of the FFT grid size. "
            "Smaller values increase focal-plane sampling at the expense of "
            "undersampling the pupil, while larger values improve pupil sampling "
            "but reduce the field of view and sampling resolution of the computed PSF."
        ),
    ).tag(config=True)

    noll_max = Int(
        default_value=11,
        help="Highest Noll index included",
    )

    wavelength_samples = Int(
        default_value=20,
        help="Number of wavelength samples for polychromatic averaging",
    ).tag(config=True)

    pupil_edge_softness = Float(
        default_value=0.08,
        help=(
            "Width of the sigmoid taper applied to the pupil edge in normalized "
            "pupil-radius units. Larger values suppress diffraction ringing but "
            "slightly blur the effective aperture."
        ),
    ).tag(config=True)

    focal_plane_smoothing_sigma_pix = Float(
        default_value=3.0,
        help="Gaussian smoothing sigma applied to PSF intensity.",
    ).tag(config=True)

    cherenkov_spectrum_index = Float(
        default_value=2.0,
        help="Power-law index for Cherenkov spectrum weighting (dN/dλ ∝ λ^-index)",
    ).tag(config=True)

    # Universal physical constants
    wavelength_min = AstroQuantity(
        default_value=350e-9 * u.m,
        physical_type=u.physical.length,
        help="Minimum wavelength for polychromatic averaging",
    ).tag(config=True)

    wavelength_max = AstroQuantity(
        default_value=550e-9 * u.m,
        physical_type=u.physical.length,
        help="Maximum wavelength for polychromatic averaging",
    ).tag(config=True)

    # Per-telescope optical parameters
    psf_reference = TelescopeParameter(
        trait=AstroQuantity(physical_type=u.physical.angle),
        default_value=0.24 * u.deg,
        help=(
            "Angular width of the square grid used to represent a single "
            "point source's PSF, centered on the source position. Must be "
            "large enough to contain the full extent of the PSF (including "
            "aberration tails) or normalization will be biased low; "
            "not the telescope's camera field of view."
        ),
    ).tag(config=True)

    z2 = TelescopeParameter(
        trait=AstroQuantity(physical_type=u.physical.length),
        default_value=0.0 * u.m,
        help="Tilt X",
    ).tag(config=True)
    z3 = TelescopeParameter(
        trait=AstroQuantity(physical_type=u.physical.length),
        default_value=0.0 * u.m,
        help="Tilt Y",
    ).tag(config=True)
    z4 = TelescopeParameter(
        trait=AstroQuantity(physical_type=u.physical.length),
        default_value=1.013e-07 * u.m,
        help="Defocus",
    ).tag(config=True)
    z5 = TelescopeParameter(
        trait=AstroQuantity(physical_type=u.physical.length),
        default_value=0.0 * u.m,
        help="Astigmatism 45°",
    ).tag(config=True)
    z6 = TelescopeParameter(
        trait=AstroQuantity(physical_type=u.physical.length),
        default_value=0.0 * u.m,
        help="Astigmatism 0°",
    ).tag(config=True)
    z7 = TelescopeParameter(
        trait=AstroQuantity(physical_type=u.physical.length),
        default_value=0.0 * u.m,
        help="Coma X base",
    ).tag(config=True)
    z8 = TelescopeParameter(
        trait=AstroQuantity(physical_type=u.physical.length),
        default_value=0.0 * u.m,
        help="Coma Y base",
    ).tag(config=True)
    z9 = TelescopeParameter(
        trait=AstroQuantity(physical_type=u.physical.length),
        default_value=0.0 * u.m,
        help="Trefoil X",
    ).tag(config=True)
    z10 = TelescopeParameter(
        trait=AstroQuantity(physical_type=u.physical.length),
        default_value=0.0 * u.m,
        help="Trefoil Y",
    ).tag(config=True)
    z11 = TelescopeParameter(
        trait=AstroQuantity(physical_type=u.physical.length),
        default_value=3.648e-08 * u.m,
        help="Spherical",
    ).tag(config=True)

    # Composite units (length/angle) don't map onto one of astropy's named
    # physical types, so `physical_type` is derived from the unit itself
    # rather than a named constant like u.physical.length/angle.
    z7_theta = TelescopeParameter(
        trait=AstroQuantity(physical_type=(u.m / u.deg).physical_type),
        default_value=2.332e-08 * u.m / u.deg,
        help="Linear coma growth",
    ).tag(config=True)

    z8_theta = TelescopeParameter(
        trait=AstroQuantity(physical_type=(u.m / u.deg).physical_type),
        default_value=1.919e-07 * u.m / u.deg,
        help="Linear coma growth",
    ).tag(config=True)

    z5_theta2 = TelescopeParameter(
        trait=AstroQuantity(physical_type=(u.m / u.deg**2).physical_type),
        default_value=7.913e-08 * u.m / u.deg**2,
        help="Quadratic astigmatism growth",
    ).tag(config=True)

    z6_theta2 = TelescopeParameter(
        trait=AstroQuantity(physical_type=(u.m / u.deg**2).physical_type),
        default_value=2.397e-08 * u.m / u.deg**2,
        help="Quadratic astigmatism growth",
    ).tag(config=True)

    @cached_property
    def _radial_order(self):
        n = 0
        while (n + 1) * (n + 2) // 2 < self.noll_max:
            n += 1
        return max(1, n)

    @cached_property
    def _zernike_grid(self):
        n = self.pupil_size
        frac = self.pupil_diameter_fraction
        if not (0 < frac <= 1):
            raise ValueError("pupil_diameter_fraction must be in (0, 1]")

        coord_limit = 1.0 / frac
        x = np.linspace(-coord_limit, coord_limit, n)
        y = np.linspace(-coord_limit, coord_limit, n)
        xx, yy = np.meshgrid(x, y)

        rr = np.sqrt(xx**2 + yy**2)
        mask = rr <= 1

        edge = max(self.pupil_edge_softness, 1e-6)
        aperture = 1.0 / (1.0 + np.exp(np.clip((rr - 1.0) / edge, -60.0, 60.0)))
        rz = RZern(self._radial_order)
        rz.make_cart_grid(xx, yy)

        return rz, mask, aperture

    def _coeff_vector(self, tel_id, lon0_deg, lat0_deg):
        """
        Build the Noll coefficient vector [m] for a given telescope and
        field-of-view offset. Internally works with plain floats in fixed
        units (m for OPD amplitudes, deg for angles) after unwrapping the
        Quantity-valued traits, since the downstream Zernike/FFT machinery
        is unit-agnostic.
        """
        rz, _, _ = self._zernike_grid
        coeff = np.zeros(rz.nk)

        theta2 = lon0_deg**2 + lat0_deg**2
        theta = np.sqrt(theta2)
        if theta > 0:
            ux = lon0_deg / theta
            uy = lat0_deg / theta
        else:
            ux = 0.0
            uy = 0.0

        z7_theta_m_per_deg = self.z7_theta.tel[tel_id].to_value(u.m / u.deg)
        z8_theta_m_per_deg = self.z8_theta.tel[tel_id].to_value(u.m / u.deg)
        z5_theta2_m_per_deg2 = self.z5_theta2.tel[tel_id].to_value(u.m / u.deg**2)
        z6_theta2_m_per_deg2 = self.z6_theta2.tel[tel_id].to_value(u.m / u.deg**2)

        coma_radial = -z7_theta_m_per_deg * theta
        coma_tangential = z8_theta_m_per_deg * theta
        coma_x = coma_radial * ux - coma_tangential * uy
        coma_y = coma_radial * uy + coma_tangential * ux
        noll_coeffs = [
            0.0,
            self.z2.tel[tel_id].to_value(u.m),
            self.z3.tel[tel_id].to_value(u.m),
            self.z4.tel[tel_id].to_value(u.m),
            self.z5.tel[tel_id].to_value(u.m) + z5_theta2_m_per_deg2 * theta2,
            self.z6.tel[tel_id].to_value(u.m) + z6_theta2_m_per_deg2 * theta2,
            self.z7.tel[tel_id].to_value(u.m) + coma_x,
            self.z8.tel[tel_id].to_value(u.m) + coma_y,
            self.z9.tel[tel_id].to_value(u.m),
            self.z10.tel[tel_id].to_value(u.m),
            self.z11.tel[tel_id].to_value(u.m),
        ]

        upper = min(len(noll_coeffs), self.noll_max, rz.nk)
        coeff[1:upper] = noll_coeffs[1:upper]
        return coeff

    def _build_psf(self, tel_id, lon0_deg, lat0_deg):
        rz, mask, aperture = self._zernike_grid
        coeff = self._coeff_vector(tel_id, lon0_deg, lat0_deg)
        wavefront = rz.eval_grid(coeff, matrix=True)

        lam_min = self.wavelength_min.to_value(u.m)
        lam_max = self.wavelength_max.to_value(u.m)
        if self.wavelength_samples < 1:
            raise ValueError("wavelength_samples must be >= 1")
        if lam_min <= 0 or lam_max <= 0 or lam_max < lam_min:
            raise ValueError("Invalid wavelength_min/wavelength_max")

        wavelengths = np.linspace(lam_min, lam_max, self.wavelength_samples)
        weights = wavelengths ** (-self.cherenkov_spectrum_index)
        weights /= np.sum(weights)
        intensity = np.zeros_like(wavefront, dtype=float)

        for wavelength, weight in zip(wavelengths, weights):
            phase = 2 * np.pi * wavefront / wavelength
            phase_safe = np.zeros_like(phase)
            phase_safe[mask] = phase[mask]
            pupil = aperture * np.exp(1j * phase_safe)
            field = fft2(pupil)
            intensity += weight * fftshift(np.abs(field) ** 2)

        sigma_pix = max(self.focal_plane_smoothing_sigma_pix, 0.0)
        if sigma_pix > 0:
            intensity = gaussian_filter(intensity, sigma=sigma_pix, mode="nearest")

        # Convert from discrete probability per FFT pixel to
        # probability density per angular area.
        total = intensity.sum()
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError(
                f"Invalid PSF normalization: total intensity is {total!r}"
            )
        intensity /= total

        psf_reference = self.psf_reference.tel[tel_id].to_value(u.deg)
        pixel_scale = psf_reference / (self.pupil_size - 1)
        pixel_area = pixel_scale**2

        intensity /= pixel_area

        return intensity

    @u.quantity_input(
        lon=u.deg,
        lat=u.deg,
        lon0=u.deg,
        lat0=u.deg,
    )
    def pdf(self, tel_id, lon, lat, lon0, lat0):
        dx = np.asarray((lon - lon0).to_value(u.deg))
        dy = np.asarray((lat - lat0).to_value(u.deg))
        input_shape = dx.shape

        lon0_deg = lon0.to_value(u.deg)
        lat0_deg = lat0.to_value(u.deg)
        intensity = self._build_psf(tel_id, lon0_deg, lat0_deg)

        full_field = self.psf_reference.tel[tel_id].to_value(u.deg)
        half_field = 0.5 * full_field
        if half_field <= 0:
            raise ValueError("psf_reference must be > 0")

        n = self.pupil_size
        xpix = (dx + half_field) / (2 * half_field) * (n - 1)
        ypix = (-dy + half_field) / (2 * half_field) * (n - 1)

        # map_coordinates requires the "points" dimension to have rank >= 1;
        # scalar lon/lat collapse to 0-d, so flatten for the call and restore
        # the original shape (including scalar) afterward.
        coords = np.array([np.atleast_1d(ypix).ravel(), np.atleast_1d(xpix).ravel()])

        psf = map_coordinates(
            intensity,
            coords,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

        psf = np.asarray(psf, dtype=float).reshape(input_shape)
        psf = np.clip(psf, 0.0, None)

        return psf.item() if psf.shape == () else psf
