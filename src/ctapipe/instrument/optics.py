"""
Classes and functions related to telescope Optics
"""

import logging
from abc import abstractmethod
from enum import Enum, auto, unique

import astropy.units as u
import numpy as np
from astropy.table import QTable
from scipy.stats import laplace, laplace_asymmetric

from ctapipe.core.component import non_abstract_children

from ..compat import StrEnum
from ..utils import get_table_dataset
from .warnings import warn_from_name

logger = logging.getLogger(__name__)

__all__ = [
    "OpticsDescription",
    "FocalLengthKind",
    "PSFModel",
    "ComaModel",
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


class PSFModel:
    def __init__(self, **kwargs):
        """
        Base component to describe image distortion due to the optics of the different cameras.
        """

    @classmethod
    def subclass_from_name(cls, name, **kwargs):
        """
        Obtain an instance of a subclass via its name

        Parameters
        ----------
        name : str
            Name of the subclass to obtain

        Returns
        -------
        Instance
            Instance of subclass to this class
        """

        subclasses = {base.__name__: base for base in non_abstract_children(cls)}

        requested_subclass = subclasses[name]
        return requested_subclass(**kwargs)

    @abstractmethod
    def pdf(self, *args):
        pass

    @abstractmethod
    def update_location(self, *args):
        pass

    @abstractmethod
    def update_model_parameters(self, *args):
        pass


class ComaModel(PSFModel):
    """
    PSF model, describing pure coma aberrations PSF effect
    """

    def __init__(
        self,
        asymmetry_params=[0.49244797, 9.23573115, 0.15216096],
        radial_scale_params=[0.01409259, 0.02947208, 0.06000271, -0.02969355],
        az_scale_params=[0.24271557, 7.5511501, 0.02037972],
    ):
        """
        PSF model, describing purely the effect of coma aberration on the PSF
        for reference see :cite:p:`startracker`
        uses an asymmetric laplacian for the radial part

        .. math:: f_{R}(r, K) = \begin{cases}\frac{1}{S_{R}(K+K^{-1})}e^{-K\frac{r-L}{S_{R}}}, r\ge L\\ \frac{1}{S_{R}(K+K^{-1})}e^{\frac{r-L}{KS_{R}}}, r < L\end{cases}

        and a symmetric laplacian in azimuthal direction

        .. math:: f_{\Phi}(\phi) = \frac{1}{2S_\phi}e^{-|\frac{\phi-\phi_0}{S_\phi}|}

        Parameters
        ----------
        asymmetry_params: list
            Parameters describing the dependency of the asymmetry of the psf on the distance to the center of the camera
            Used to calculate a pdf asymmetry parameter K of the asymmetric radial laplacian of the psf as a function of the distance r to the optical axis

            .. math:: K(r) = 1 - asym_0 \tanh(asym_1 r) - asym_2 r

        radial_scale_params : list
            Parameters describing the dependency of the radial scale on the distance to the center of the camera
            Used to calculate width Sr of the asymmetric radial laplacian in the PSF as a of function the distance r to the optical axis

            .. math:: S_{R}(r) & = b_1 - b_2\,r + b_3\,r^2 + b_4\,r^3

        az_scale_params : list
            Parameters describing the dependency of the azimuthal scale on the distance to the center of the camera
            Used to calculate width Sf of the azimuthal laplacian in the PSF as a function of the angle phi

            .. math:: S_{\phi}(r) & = a_1\,\exp{(-a_2\,r)}+\frac{a_3}{a_3+r}

        """
        if (
            len(asymmetry_params) == 3
            and len(radial_scale_params) == 4
            and len(az_scale_params) == 3
        ):
            self.asymmetry_params = asymmetry_params
            self.radial_scale_params = radial_scale_params
            self.az_scale_params = az_scale_params
        else:
            raise ValueError(
                "asymmetry_params and az_scale_params needs to have length 3 and radial_scale_params length 4"
            )

        self.update_location(0.0, 0.0)

    def k_func(self, x):
        return (
            1
            - self.asymmetry_params[0] * np.tanh(self.asymmetry_params[1] * x)
            - self.asymmetry_params[2] * x
        )

    def sr_func(self, x):
        return np.polyval(self.radial_scale_params, x)

    def sf_func(self, x):
        return self.az_scale_params[0] * np.exp(
            -self.az_scale_params[1] * x
        ) + self.az_scale_params[2] / (self.az_scale_params[2] + x)

    def pdf(self, r, f, r0, f0):
        """
        Calculates the value of the psf at a given location

        Parameters
        ----------
        r : float
            distance to the center of the camera in meters, location at where the PSF is evaluated
        f : float
            polar angle in radians, location at where the PSF is evaluated
        r0 : float
            distance to the center of the camera in meters, location from where the PSF is evaluated
        f0 : float
            polar angle in radians, location from where the PSF is evaluated
        Returns
        ----------
        psf : float
            value of the PSF at the specified location
        """
        k = self.k_func(r0)
        sr = self.sr_func(r0)
        sf = self.sf_func(r0)
        self.radial_pdf_params = (k, r0, sr)
        self.azimuthal_pdf_params = (f0, sf)

        return laplace_asymmetric.pdf(r, *self.radial_pdf_params) * laplace.pdf(
            f, *self.azimuthal_pdf_params
        )

    def update_model_parameters(self, model_params):
        """
        Updates the model parameters for the psf

        Parameters
        ----------
        model_params : dict
            dictionary with the model parameters
            needs to have the keys asymmetry_params, radial_scale_params and az_scale_params
            The values need to be lists of length 3, 4 and 3 respectively
        """
        if not (
            len(model_params["asymmetry_params"]) == 3
            and len(model_params["radial_scale_params"]) == 4
            and len(model_params["az_scale_params"]) == 3
        ):
            raise ValueError(
                "asymmetry_params and az_scale_params needs to have length 3 and radial_scale_params length 4"
            )

        self.asymmetry_params = model_params["asymmetry_params"]
        self.radial_scale_params = model_params["radial_scale_params"]
        self.az_scale_params = model_params["az_scale_params"]
        k = self.k_func(self.radial_pdf_params[1])
        sr = self.sr_func(self.radial_pdf_params[1])
        sf = self.sf_func(self.radial_pdf_params[1])
        self.radial_pdf_params = (k, self.radial_pdf_params[1], sr)
        self.azimuthal_pdf_params = (self.azimuthal_pdf_params[0], sf)
