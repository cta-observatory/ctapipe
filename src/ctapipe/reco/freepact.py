"""FreePACT Reconstructor for ctapipe
This module implements the FreePACT reconstructor, which is a subclass of ImPACTReconstructor.
It uses a neural network to predict th likelihood camera images based on the shower parameters."""

import numpy as np
import numpy.ma as ma

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

from ctapipe.core import traits
from ctapipe.core.telescope_component import TelescopeParameter
from ctapipe.reco.impact import ImPACTReconstructor
from ctapipe.utils.template_network_interpolator import FreePACTInterpolator

from ..exceptions import OptionalDependencyMissing
from .impact_utilities import rotate_translate

__all__ = ["FreePACTReconstructor", "create_dummy_freepact_templates"]


class FreePACTReconstructor(ImPACTReconstructor):
    """
    This class implements the FreePACT reconstructor, which uses a neural network
    to predict the likelihood of camera images based on shower parameters. The
    reconstructor is based on the work presented in schwefer24.

    Because this application is computationally intensive the usual
    advice to use astropy units for all quantities is ignored (as
    these slow down some computations), instead units within the class
    are fixed:

    - Angular units in radians
    - Distance units in metres
    - Energy units in TeV

    Parameters
    ----------
    subarray : ctapipe.instrument.SubarrayDescription
        The telescope subarray to use for reconstruction
    atmosphere_profile : ctapipe.atmosphere.AtmosphereDensityProfile
        Density vs. altitude profile of the local atmosphere

    References
    ----------
    .. [schwefer24] Schwefer, Parsons, & Hinton,  Astroparticle Physics 163 (2024), 103008
    """

    image_template_path = TelescopeParameter(
        trait=traits.Path(
            exists=True, directory_ok=True, allow_none=False, default_value="./"
        ),
        allow_none=False,
        help=("Path to the image templates to be used in the reconstruction"),
    ).tag(config=True)

    def set_up_templates(self):
        """Set up the templates for the FreePACT reconstructor."""
        template_sort_dict = {}

        for tel_id in self.subarray.tel_ids:
            if self.image_template_path.tel[tel_id] not in template_sort_dict.keys():
                template_sort_dict[self.image_template_path.tel[tel_id]] = [tel_id]
            else:
                template_sort_dict[self.image_template_path.tel[tel_id]].append(tel_id)

        for template_path, tel_ids in template_sort_dict.items():
            net_interpolator = FreePACTInterpolator(template_path)

            self.prediction[tuple(tel_ids)] = net_interpolator

    def get_likelihood(
        self,
        source_x,
        source_y,
        core_x,
        core_y,
        energy,
        x_max_scale,
        goodness_of_fit=False,
    ):
        """Get the likelihood that the image predicted at the given test
        position matches the camera image.

        Parameters
        ----------
        source_x: float
            Source position of shower in the nominal system (in deg)
        source_y: float
            Source position of shower in the nominal system (in deg)
        core_x: float
            Core position of shower in tilted telescope system (in m)
        core_y: float
            Core position of shower in tilted telescope system (in m)
        energy: float
            Shower energy (in TeV)
        x_max_scale: float
            Scaling factor applied to geometrically calculated Xmax

        Returns
        -------
        float: Likelihood the model represents the camera image at this position

        """
        if np.isnan(source_x) or np.isnan(source_y):
            return 1e8

        # First we add units back onto everything.  Currently not
        # handled very well, maybe in future we could just put
        # everything in the correct units when loading in the class
        # and ignore them from then on

        zenith = self.zenith
        azimuth = self.azimuth

        # Geometrically calculate the depth of maximum given this test position
        x_max_guess = self.get_shower_max(source_x, source_y, core_x, core_y, zenith)
        x_max_guess *= x_max_scale

        # Convert to binning of Xmax
        x_max_diff = x_max_guess

        # Calculate impact distance for all telescopes
        impact = np.sqrt(
            (self.tel_pos_x - core_x) ** 2 + (self.tel_pos_y - core_y) ** 2
        )
        # And the expected rotation angle
        phi = np.arctan2(self.tel_pos_y - core_y, self.tel_pos_x - core_x)

        # Rotate and translate all pixels such that they match the
        # template orientation
        # numba does not support masked arrays, work on underlying array and add mask back
        pix_x_rot, pix_y_rot = rotate_translate(
            self.pixel_y.data, self.pixel_x.data, source_y, source_x, -phi
        )
        pix_x_rot = np.ma.array(pix_x_rot, mask=self.pixel_x.mask)
        pix_y_rot = np.ma.array(pix_y_rot, mask=self.pixel_y.mask)
        # In the interpolator class we can gain speed advantages by using masked arrays
        # so we need to make sure here everything is masked
        likelihood = ma.zeros(self.image.shape)
        likelihood.mask = ma.getmask(self.image)

        for tel_ids, template in self.prediction.items():
            template_mask = self.template_masks[tel_ids]
            if np.any(template_mask):
                likelihood[template_mask] = template(
                    np.rad2deg(zenith),
                    azimuth,
                    energy * np.ones_like(impact[template_mask]),
                    impact[template_mask],
                    x_max_diff * np.ones_like(impact[template_mask]),
                    np.rad2deg(pix_x_rot[template_mask]),
                    np.rad2deg(pix_y_rot[template_mask]),
                    self.image[template_mask],
                )
        likelihood.mask = ma.getmask(self.image)

        if goodness_of_fit:
            return -2 * ma.sum(likelihood) / np.sum(~ma.getmask(self.image))

        likelihood = ma.sum(likelihood) * -2
        return likelihood


class FreePACTProtonReconstructor(FreePACTReconstructor):
    """
    This class implements the FreePACT reconstructor for proton showers.
    It uses a neural network to predict the likelihood of camera images based on shower parameters.
    """

    image_template_path = TelescopeParameter(
        trait=traits.Path(
            exists=True, directory_ok=True, allow_none=False, default_value="./"
        ),
        allow_none=False,
        help=("Path to the image templates to be used in the reconstruction"),
    ).tag(config=True)


def create_dummy_freepact_templates(
    output_dir, telescope_type, zenith, azimuth, offset
):
    """Create file with dummy template library

    Args:
        output_dir (str): Output directory
        telescope_type (str): Telescope type
        zenith (float): Zenith angle in radians
        azimuth (float): Azimuth angle in radians
        offset (float): Offset in degrees
    """

    if tf is None:
        raise OptionalDependencyMissing("tensorflow")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(6,)),
            tf.keras.layers.Dense(5, activation="sigmoid", kernel_initializer="zeros"),
            tf.keras.layers.Dense(5, activation="sigmoid", kernel_initializer="zeros"),
            tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer="zeros"),
        ]
    )

    model.compile(optimizer="adam", loss="mse")

    # reformat line to create file name string
    output_file = f"/predict_{telescope_type}_{int(zenith)}deg_{int(azimuth)}deg_{offset:.1f}off.keras"

    model.save(output_dir + output_file)
