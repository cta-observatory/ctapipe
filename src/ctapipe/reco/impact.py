#!/usr/bin/env python3
"""
Implementation of the ImPACT reconstruction algorithm
"""
import copy
from string import Template

import numpy as np
import numpy.ma as ma
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from scipy.stats import norm

from ctapipe.core import traits
from ctapipe.exceptions import OptionalDependencyMissing

from ..compat import COPY_IF_NEEDED
from ..containers import ReconstructedEnergyContainer, ReconstructedGeometryContainer
from ..coordinates import (
    CameraFrame,
    GroundFrame,
    NominalFrame,
    TiltedGroundFrame,
    project_to_ground,
)
from ..core import Provenance
from ..fitting import lts_linear_regression
from ..image.cleaning import dilate
from ..image.pixel_likelihood import (
    mean_poisson_likelihood_gaussian,
    neg_log_likelihood_approx,
)
from ..utils.template_network_interpolator import (
    DummyTemplateInterpolator,
    DummyTimeInterpolator,
    TemplateNetworkInterpolator,
    TimeGradientInterpolator,
)
from .impact_utilities import (
    EmptyImages,
    create_seed,
    guess_shower_depth,
    rotate_translate,
)
from .reconstructor import (
    HillasGeometryReconstructor,
    InvalidWidthException,
    ReconstructionProperty,
    TooFewTelescopesException,
)

try:
    from iminuit import Minuit
except ModuleNotFoundError:
    Minuit = None

PROV = Provenance()

INVALID_GEOMETRY = ReconstructedGeometryContainer(
    telescopes=[],
    prefix="ImPACTReconstructor",
)

INVALID_ENERGY = ReconstructedEnergyContainer(
    prefix="ImPACTReconstructor",
    telescopes=[],
)

# These are settings for the iminuit minimizer
MINUIT_ERRORDEF = 0.5  # 0.5 for a log-likelihood cost function for correct errors
MINUIT_STRATEGY = 1  # Default minimization strategy, 2 is careful, 0 is fast
MINUIT_TOLERANCE_FACTOR = 1000  # Tolerance for convergence according to EDM criterion
MIGRAD_ITERATE = 1  # Do not call migrad again if convergence was not reached
__all__ = ["ImPACTReconstructor"]


class ImPACTReconstructor(HillasGeometryReconstructor):
    """This class is an implementation if the impact_reco Monte Carlo
    Template based image fitting method from parsons14.  This method uses a
    comparison of the predicted image from a library of image
    templates to perform a maximum likelihood fit for the shower axis,
    energy and height of maximum.

    Besides the image information, there is also the option to use the time gradient of the pixels
    across the image as additional information in the fit. This requires an additional set of templates

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
    dummy_reconstructor : bool, optional
        Option to use a set of dummy templates. This can be used for testing the algorithm,
        but for any actual reconstruction should be set to its default False

    References
    ----------
    .. [parsons14] Parsons & Hinton, Astroparticle Physics 56 (2014), pp. 26-34

    """

    use_time_gradient = traits.Bool(
        default_value=False,
        help="Use time gradient in ImPACT reconstruction. Requires an extra set of time gradient templates",
    ).tag(config=True)

    root_dir = traits.Unicode(
        default_value=".", help="Directory containing ImPACT tables"
    ).tag(config=True)

    # For likelihood calculation we need the with of the
    # pedestal distribution for each pixel
    # currently this is not available from the calibration,
    # so for now lets hard code it in a dict
    ped_table = {
        "LSTCam": 1.4,
        "NectarCam": 1.3,
        "FlashCam": 1.3,
        "SST-Camera": 0.5,
        "CHEC": 0.5,
        "ASTRICam": 0.5,
        "dummy": 0.01,
        "UNKNOWN-960PX": 1.0,
    }
    spe = 0.6  # Also hard code single p.e. distribution width

    property = ReconstructionProperty.ENERGY | ReconstructionProperty.GEOMETRY

    def __init__(
        self, subarray, atmosphere_profile, dummy_reconstructor=False, **kwargs
    ):
        if Minuit is None:
            raise OptionalDependencyMissing("iminuit") from None

        if atmosphere_profile is None:
            raise TypeError(
                "Argument 'atmosphere_profile' can not be 'None' for ImPACTReconstructor"
            )

        super().__init__(subarray, atmosphere_profile, **kwargs)

        # First we create a dictionary of image template interpolators
        # for each telescope type
        # self.priors = prior

        # String templates for loading ImPACT templates
        self.amplitude_template = Template("${base}/${camera}.template.gz")
        self.time_template = Template("${base}/${camera}_time.template.gz")

        # Next we need the position, area and amplitude from each pixel in the event
        # making this a class member makes passing them around much easier

        self.pixel_x, self.pixel_y = None, None
        self.image, self.time = None, None

        self.tel_types, self.tel_id = None, None

        # We also need telescope positions
        self.tel_pos_x, self.tel_pos_y = None, None

        # And the peak of the images
        self.peak_x, self.peak_y, self.peak_amp = None, None, None
        self.hillas_parameters, self.ped = None, None

        self.prediction = dict()
        self.time_prediction = dict()

        self.array_direction = None
        self.nominal_frame = None

        self.dummy_reconstructor = dummy_reconstructor

    def __call__(self, event):
        """
        Perform the full shower geometry reconstruction on the input event.

        Parameters
        ----------
        event : container
            `ctapipe.containers.ArrayEventContainer`
        """

        try:
            hillas_dict = self._create_hillas_dict(event)
        except (TooFewTelescopesException, InvalidWidthException):
            event.dl2.stereo.geometry[self.__class__.__name__] = INVALID_GEOMETRY
            event.dl2.stereo.energy[self.__class__.__name__] = INVALID_ENERGY
            self._store_impact_parameter(event)
            return

        # Due to tracking the pointing of the array will never be a constant
        array_pointing = SkyCoord(
            az=event.pointing.array_azimuth,
            alt=event.pointing.array_altitude,
            frame=AltAz(),
        )

        # And the pointing direction of the telescopes may not be the same
        telescope_pointings = self._get_telescope_pointings(event)

        # Finally get the telescope images and and the selection masks
        mask_dict, image_dict, time_dict = {}, {}, {}
        for tel_id in hillas_dict.keys():
            image = event.dl1.tel[tel_id].image
            image_dict[tel_id] = image
            time_dict[tel_id] = event.dl1.tel[tel_id].peak_time
            mask = event.dl1.tel[tel_id].image_mask

            # Dilate the images around the original cleaning to help the fit
            for _ in range(3):
                mask = dilate(self.subarray.tel[tel_id].camera.geometry, mask)

            mask_dict[tel_id] = mask

        # Next, we look for geometry and energy seeds from previously applied reconstructors.
        # Both need to be present at elast once for ImPACT to run.

        reco_geom_pred = event.dl2.stereo.geometry

        valid_geometry_seed = False
        for geom_pred in reco_geom_pred.values():
            if geom_pred.is_valid:
                valid_geometry_seed = True
                break

        reco_energy_pred = event.dl2.stereo.energy

        valid_energy_seed = False
        for E_pred in reco_energy_pred.values():
            if E_pred.is_valid:
                valid_energy_seed = True
                break

        if valid_geometry_seed is False or valid_energy_seed is False:
            event.dl2.stereo.geometry[self.__class__.__name__] = INVALID_GEOMETRY
            event.dl2.stereo.energy[self.__class__.__name__] = INVALID_ENERGY
            self._store_impact_parameter(event)
            return

        shower_result, energy_result = self.predict(
            hillas_dict=hillas_dict,
            subarray=self.subarray,
            shower_seed=reco_geom_pred,
            energy_seed=reco_energy_pred,
            array_pointing=array_pointing,
            telescope_pointings=telescope_pointings,
            image_dict=image_dict,
            mask_dict=mask_dict,
            time_dict=time_dict,
        )
        event.dl2.stereo.geometry[self.__class__.__name__] = shower_result
        event.dl2.stereo.energy[self.__class__.__name__] = energy_result

        self._store_impact_parameter(event)

    def initialise_templates(self, tel_type):
        """Check if templates for a given telescope type has been initialised
        and if not do it and add to the dictionary

        Parameters
        ----------
        tel_type: dictionary
            Dictionary of telescope types in event

        Returns
        -------
        boolean: Confirm initialisation

        """

        for t in tel_type:
            if tel_type[t] in self.prediction.keys() or tel_type[t] == "DUMMY":
                continue

            if self.dummy_reconstructor:
                self.prediction[tel_type[t]] = DummyTemplateInterpolator()
            else:
                filename = self.amplitude_template.substitute(
                    base=self.root_dir, camera=tel_type[t]
                )
                self.prediction[tel_type[t]] = TemplateNetworkInterpolator(
                    filename, bounds=((-5, 1), (-1.5, 1.5))
                )
                PROV.add_input_file(
                    filename, role="ImPACT Template file for " + tel_type[t]
                )

            if self.use_time_gradient:
                if self.dummy_reconstructor:
                    self.time_prediction[tel_type[t]] = DummyTimeInterpolator()
                else:
                    filename = self.time_template.substitute(
                        base=self.root_dir, camera=tel_type[t]
                    )
                    self.time_prediction[tel_type[t]] = TimeGradientInterpolator(
                        filename
                    )
                    PROV.add_input_file(
                        filename, role="ImPACT Time Template file for " + tel_type[t]
                    )

        return True

    def get_hillas_mean(self):
        """This is a simple function to find the peak position of each image
        in an event which will be used later in the Xmax calculation. Peak is
        found by taking the average position of the n hottest pixels in the
        image.
        """
        peak_x = np.zeros([len(self.pixel_x)])  # Create blank arrays for peaks
        # rather than a dict (faster)
        peak_y = np.zeros(peak_x.shape)
        peak_amp = np.zeros(peak_x.shape)

        # Loop over all tels to take weighted average of pixel
        # positions This loop could maybe be replaced by an array
        # operation by a numpy wizard
        # Maybe a vectorize?
        tel_num = 0

        for hillas in self.hillas_parameters:
            peak_x[tel_num] = hillas.fov_lon.to(u.rad).value  # Fill up array
            peak_y[tel_num] = hillas.fov_lat.to(u.rad).value
            peak_amp[tel_num] = hillas.intensity
            tel_num += 1

        self.peak_x = peak_x  # * unit # Add to class member
        self.peak_y = peak_y  # * unit
        self.peak_amp = peak_amp

    # This function would be useful elsewhere so probably be implemented in a
    # more general form
    def get_shower_max(self, source_x, source_y, core_x, core_y, zen):
        """Function to calculate the depth of shower maximum geometrically
        under the assumption that the shower maximum lies at the
        brightest point of the camera image.

        Parameters
        ----------
        source_x: float
            Event source position in nominal frame
        source_y: float
            Event source position in nominal frame
        core_x: float
            Event core position in telescope tilted frame
        core_y: float
            Event core position in telescope tilted frame
        zen: float
            Zenith angle of event

        Returns
        -------
        float: Depth of maximum of air shower

        """
        # Calculate displacement of image centroid from source position (in
        # rad)
        disp = np.sqrt((self.peak_x - source_x) ** 2 + (self.peak_y - source_y) ** 2)

        # Calculate impact parameter of the shower
        impact = np.sqrt(
            (self.tel_pos_x - core_x) ** 2 + (self.tel_pos_y - core_y) ** 2
        )
        # Distance above telescope is ratio of these two (small angle)

        height_tilted = (
            impact / disp
        )  # This is in tilted frame along the telescope axis
        weight = np.power(self.peak_amp, 0.0)  # weight average by sqrt amplitude
        # sqrt may not be the best option...

        # Take weighted mean of estimates, converted to height above ground
        mean_height_above_ground = np.sum(
            height_tilted * np.cos(zen) * weight
        ) / np.sum(weight)

        # Add on the height of the detector above sea level
        mean_height_asl = (
            mean_height_above_ground
            + self.subarray.reference_location.height.to_value(u.m)
        )

        if mean_height_asl > 100000 or np.isnan(mean_height_asl):
            mean_height_asl = 100000

        # Lookup this height in the depth tables, the convert Hmax to Xmax
        slant_depth = self.atmosphere_profile.slant_depth_from_height(
            mean_height_asl * u.m, zen * u.rad
        )

        return slant_depth.to_value(u.g / (u.cm * u.cm))

    def image_prediction(
        self, tel_type, zenith, azimuth, energy, impact, x_max, pix_x, pix_y
    ):
        """Creates predicted image for the specified pixels, interpolated
        from the template library.

        Parameters
        ----------
        tel_type: string
            Telescope type specifier
        energy: float
            Event energy (TeV)
        impact: float
            Impact diance of shower (metres)
        x_max: float
            Depth of shower maximum (num bins from expectation)
        pix_x: ndarray
            X coordinate of pixels
        pix_y: ndarray
            Y coordinate of pixels

        Returns
        -------
        ndarray: predicted amplitude for all pixels

        """
        return self.prediction[tel_type](
            zenith, azimuth, energy, impact, x_max, pix_x, pix_y
        )

    def predict_time(self, tel_type, zenith, azimuth, energy, impact, x_max):
        """Creates predicted image for the specified pixels, interpolated
        from the template library.

        Parameters
        ----------
        tel_type: string
            Telescope type specifier
        energy: float
            Event energy (TeV)
        impact: float
            Impact diance of shower (metres)
        x_max: float
            Depth of shower maximum (num bins from expectation)

        Returns
        -------
        ndarray: predicted amplitude for all pixels

        """
        time = self.time_prediction[tel_type](zenith, azimuth, energy, impact, x_max)
        return time.T[0], time.T[1]

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
        goodness_of_fit: boolean
            Determines whether expected likelihood should be subtracted from result
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

        # Calculate expected Xmax given this energy
        x_max_exp = guess_shower_depth(energy)  # / np.cos(20*u.deg)

        # Convert to binning of Xmax
        x_max_diff = x_max_guess - x_max_exp

        # Check for range
        if x_max_diff > 200:
            x_max_diff = 200
        if x_max_diff < -150:
            x_max_diff = -150

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
        pix_x_rot = np.ma.array(pix_x_rot, mask=self.pixel_x.mask, copy=COPY_IF_NEEDED)
        pix_y_rot = np.ma.array(pix_y_rot, mask=self.pixel_y.mask, copy=COPY_IF_NEEDED)

        # In the interpolator class we can gain speed advantages by using masked arrays
        # so we need to make sure here everything is masked
        prediction = ma.zeros(self.image.shape)
        prediction.mask = ma.getmask(self.image)

        time_gradients, time_gradients_uncertainty = (
            np.zeros(self.image.shape[0]),
            np.zeros(self.image.shape[0]),
        )
        # Loop over all telescope types and get prediction
        for tel_type in np.unique(self.tel_types).tolist():
            type_mask = self.tel_types == tel_type

            prediction[type_mask] = self.image_prediction(
                tel_type,
                np.rad2deg(zenith),
                azimuth,
                energy * np.ones_like(impact[type_mask]),
                impact[type_mask],
                x_max_diff * np.ones_like(impact[type_mask]),
                np.rad2deg(pix_x_rot[type_mask]),
                np.rad2deg(pix_y_rot[type_mask]),
            )

            if self.use_time_gradient:
                tg, tgu = self.predict_time(
                    tel_type,
                    np.rad2deg(zenith),
                    azimuth,
                    energy * np.ones_like(impact[type_mask]),
                    impact[type_mask],
                    x_max_diff * np.ones_like(impact[type_mask]),
                )
                time_gradients[type_mask] = tg
                time_gradients_uncertainty[type_mask] = tgu

        if self.use_time_gradient:
            time_gradients_uncertainty[time_gradients_uncertainty == 0] = 1e-6

            chi2 = 0
            for telescope_index, (image, time) in enumerate(zip(self.image, self.time)):
                time_mask = np.logical_and(
                    np.invert(ma.getmask(image)),
                    time > 0,
                )
                time_mask = np.logical_and(time_mask, np.isfinite(time))
                time_mask = np.logical_and(time_mask, image > 5)
                if (
                    np.sum(time_mask) > 3
                    and time_gradients_uncertainty[telescope_index] > 0
                ):
                    time_slope = lts_linear_regression(
                        x=np.rad2deg(pix_x_rot[telescope_index][time_mask]),
                        y=time[time_mask],
                        samples=3,
                    )[0][0]

                    time_like = -1 * norm.logpdf(
                        time_slope,
                        loc=time_gradients[telescope_index],
                        scale=time_gradients_uncertainty[telescope_index],
                    )

                    chi2 += time_like

        # Likelihood function will break if we find a NaN or a 0
        prediction[np.isnan(prediction)] = 1e-8
        prediction[prediction < 1e-8] = 1e-8
        # prediction *= self.scale_factor[:, np.newaxis]

        # Get likelihood that the prediction matched the camera image
        mask = ma.getmask(self.image)

        like = neg_log_likelihood_approx(self.image, prediction, self.spe, self.ped)
        like[mask] = 0

        if goodness_of_fit:
            like_expectation_gaus = mean_poisson_likelihood_gaussian(
                prediction, self.spe, self.ped
            )
            like_expectation_gaus[mask] = 0
            mask_shower = np.invert(mask)
            goodness = np.sum(2 * like - like_expectation_gaus, axis=-1) / np.sqrt(
                2 * (np.sum(mask_shower, axis=-1) - 6)
            )
            return goodness

        like = np.sum(like)

        final_sum = like
        if self.use_time_gradient:
            final_sum += chi2

        return final_sum

    def set_event_properties(
        self,
        hillas_dict,
        image_dict,
        time_dict,
        mask_dict,
        subarray,
        array_pointing,
        telescope_pointing,
    ):
        """The setter class is used to set the event properties within this
        class before minimisation can take place. This simply copies a
        bunch of useful properties to class members, so that we can
        use them later without passing all this information around.

        Parameters
        ----------
        hillas_dict: dict
            dictionary with telescope IDs as key and
            HillasParametersContainer instances as values
        image_dict: dict
            Amplitude of pixels in camera images
        time_dict: dict
            Time information per each pixel in camera images
        mask_dict: dict
            Event image masks
        subarray: dict
            Type of telescope
        array_pointing: SkyCoord[AltAz]
            Array pointing direction in the AltAz Frame
        telescope_pointing: SkyCoord[AltAz]
            Telescope pointing directions in the AltAz Frame
        Returns
        -------
        None

        """
        # First store these parameters in the class so we can use them
        # in minimisation For most values this is simply copying
        self.image = image_dict

        self.tel_pos_x = np.zeros(len(hillas_dict))
        self.tel_pos_y = np.zeros(len(hillas_dict))
        # self.scale_factor = np.zeros(len(hillas_dict))

        self.ped = np.zeros(len(hillas_dict))
        self.tel_types, self.tel_id = list(), list()

        max_pix_x = 0
        px, py, pa, pt = list(), list(), list(), list()
        self.hillas_parameters = list()

        # Get telescope positions in tilted frame
        tilted_frame = TiltedGroundFrame(pointing_direction=array_pointing)
        ground_positions = subarray.tel_coords
        grd_coord = SkyCoord(
            x=ground_positions.x,
            y=ground_positions.y,
            z=ground_positions.z,
            frame=GroundFrame(),
        )

        self.array_direction = array_pointing
        self.zenith = (np.pi / 2) - self.array_direction.alt.to(u.rad).value
        self.azimuth = self.array_direction.az.to(u.deg).value

        self.nominal_frame = NominalFrame(origin=self.array_direction)

        tilt_coord = grd_coord.transform_to(tilted_frame)
        type_tel = {}
        indices = subarray.tel_ids_to_indices(list(hillas_dict.keys()))

        # So here we must loop over the telescopes
        for tel_id, i in zip(hillas_dict, range(len(hillas_dict))):
            geometry = subarray.tel[tel_id].camera.geometry
            type = subarray.tel[tel_id].camera.name
            type_tel[tel_id] = type

            mask = mask_dict[tel_id]

            focal_length = subarray.tel[tel_id].optics.effective_focal_length
            camera_frame = CameraFrame(
                telescope_pointing=telescope_pointing[tel_id],
                focal_length=focal_length,
            )
            camera_coords = SkyCoord(
                x=geometry.pix_x[mask], y=geometry.pix_y[mask], frame=camera_frame
            )
            nominal_coords = camera_coords.transform_to(self.nominal_frame)

            px.append(nominal_coords.fov_lon.to(u.rad).value)
            if len(px[i]) > max_pix_x:
                max_pix_x = len(px[i])
            py.append(nominal_coords.fov_lat.to(u.rad).value)
            pa.append(image_dict[tel_id][mask])
            pt.append(time_dict[tel_id][mask])

            self.ped[i] = self.ped_table[type]
            self.tel_types.append(type)
            self.tel_id.append(tel_id)

            self.tel_pos_x[i] = tilt_coord[indices[i]].x.to(u.m).value
            self.tel_pos_y[i] = tilt_coord[indices[i]].y.to(u.m).value

            self.hillas_parameters.append(hillas_dict[tel_id])
            # self.scale_factor[i] = 1#self.template_scale[tel_id]

        # Most interesting stuff is now copied to the class, but to remove our requirement
        # for loops we must copy the pixel positions to an array with the length of the
        # largest image

        # First allocate everything
        shape = (len(hillas_dict), max_pix_x)
        self.pixel_x, self.pixel_y = ma.zeros(shape), ma.zeros(shape)
        self.image, self.time, self.ped, self.spe = (
            ma.zeros(shape),
            ma.zeros(shape),
            ma.zeros(shape),
            ma.zeros(shape),
        )
        self.tel_types = np.array(self.tel_types)

        # Copy everything into our masked arrays
        for i in range(len(hillas_dict)):
            array_len = len(px[i])
            self.pixel_x[i][:array_len] = px[i]
            self.pixel_y[i][:array_len] = py[i]
            self.image[i][:array_len] = pa[i]
            self.time[i][:array_len] = pt[i]
            self.ped[i][:] = self.ped_table[self.tel_types[i]]
            self.spe[i][:] = 0.5

        # Set the image mask
        mask = self.image == 0.0
        self.pixel_x[mask], self.pixel_y[mask] = ma.masked, ma.masked
        self.image[mask] = ma.masked
        self.time[mask] = ma.masked

        # Finally run some functions to get ready for the event
        self.get_hillas_mean()
        self.initialise_templates(type_tel)

    def predict(
        self,
        hillas_dict,
        subarray,
        array_pointing,
        shower_seed,
        energy_seed,
        telescope_pointings=None,
        image_dict=None,
        mask_dict=None,
        time_dict=None,
    ):
        """Predict method for the ImPACT reconstructor.
        Used to calculate the reconstructed ImPACT shower geometry and energy.

        Parameters
        ----------
        shower_seed: ReconstructedShowerContainer
            Seed shower geometry to be used in the fit
        energy_seed: ReconstructedEnergyContainer
            Seed energy to be used in fit

        Returns
        -------
        ReconstructedShowerContainer, ReconstructedEnergyContainer:
        """
        if image_dict is None:
            raise EmptyImages("Images not passed to ImPACT reconstructor")

        self.set_event_properties(
            copy.deepcopy(hillas_dict),
            image_dict,
            time_dict,
            mask_dict,
            subarray,
            array_pointing,
            telescope_pointings,
        )

        self.reset_interpolator()

        like_min = 1e9
        fit_params = None

        for cleaning in shower_seed:
            # Copy all of our seed parameters out of the shower objects
            # We need to convert the shower direction to the nominal system
            horizon_seed = SkyCoord(
                az=shower_seed[cleaning].az,
                alt=shower_seed[cleaning].alt,
                frame=AltAz(),
            )
            nominal_seed = horizon_seed.transform_to(self.nominal_frame)

            source_x = nominal_seed.fov_lon.to_value(u.rad)
            source_y = nominal_seed.fov_lat.to_value(u.rad)
            # And the core position to the tilted ground frame
            ground = SkyCoord(
                x=shower_seed[cleaning].core_x,
                y=shower_seed[cleaning].core_y,
                z=0 * u.m,
                frame=GroundFrame,
            )
            tilted = ground.transform_to(
                TiltedGroundFrame(pointing_direction=self.array_direction)
            )
            tilt_x = tilted.x.to(u.m).value
            tilt_y = tilted.y.to(u.m).value
            zenith = 90 * u.deg - self.array_direction.alt

            for energy_reco in energy_seed:
                energy = energy_seed[energy_reco].energy.value

                seed, step, limits = create_seed(
                    source_x, source_y, tilt_x, tilt_y, energy
                )

                # Perform maximum likelihood fit
                fit_params_min, errors, like = self.minimise(
                    params=seed,
                    step=step,
                    limits=limits,
                )
                #            fit_params_min, like = self.energy_guess(seed)
                if like < like_min:
                    fit_params = fit_params_min
                    like_min = like

        if fit_params is None:
            return INVALID_GEOMETRY, INVALID_ENERGY

        # Now do full minimisation
        seed = create_seed(
            fit_params[0], fit_params[1], fit_params[2], fit_params[3], fit_params[4]
        )

        fit_params, errors, like = self.minimise(
            params=seed[0],
            step=seed[1],
            limits=seed[2],
        )

        # Create a container class for reconstructed shower

        # Convert the best fits direction and core to Horizon and ground systems and
        # copy to the shower container
        nominal = SkyCoord(
            fov_lon=fit_params[0] * u.rad,
            fov_lat=fit_params[1] * u.rad,
            frame=self.nominal_frame,
        )
        horizon = nominal.transform_to(AltAz())

        shower_result = ReconstructedGeometryContainer()
        shower_result.prefix = self.__class__.__name__
        # Transform everything back to a useful system
        shower_result.alt, shower_result.az = horizon.alt, horizon.az
        tilted = SkyCoord(
            x=fit_params[2] * u.m,
            y=fit_params[3] * u.m,
            z=0 * u.m,
            frame=TiltedGroundFrame(pointing_direction=self.array_direction),
        )
        ground = project_to_ground(tilted)

        shower_result.core_x = ground.x
        shower_result.core_y = ground.y

        shower_result.core_tilted_x = tilted.x
        shower_result.core_tilted_y = tilted.y

        shower_result.core_tilted_uncert_x = errors[2] * u.m
        shower_result.core_tilted_uncert_y = errors[3] * u.m

        shower_result.is_valid = True

        # Currently no errors not available to copy NaN
        shower_result.alt_uncert = source_x * u.rad
        shower_result.az_uncert = source_y * u.rad
        # shower_result.core_uncert = np.nan

        # Copy reconstructed Xmax
        slant_depth = (
            fit_params[5]
            * self.get_shower_max(
                fit_params[0],
                fit_params[1],
                fit_params[2],
                fit_params[3],
                zenith.to(u.rad).value,
            )
            * u.g
            / (u.cm * u.cm)
        )
        # h_max really is the vertical altitude of the maximum above sea level
        shower_result.h_max = self.atmosphere_profile.height_from_slant_depth(
            slant_depth
        )
        shower_result.h_max_uncert = errors[5] * shower_result.h_max

        goodness_of_fit = self.get_likelihood(
            fit_params[0],
            fit_params[1],
            fit_params[2],
            fit_params[3],
            fit_params[4],
            fit_params[5],
            True,
        )

        shower_result.goodness_of_fit = np.sum(goodness_of_fit)
        shower_result.telescopes = self.tel_id
        shower_result.is_valid = True

        # Create a container class for reconstructed energy
        energy_result = ReconstructedEnergyContainer(
            prefix=self.__class__.__name__,
            energy=fit_params[4] * u.TeV,
            energy_uncert=errors[4] * u.TeV,
            telescopes=self.tel_id,
            is_valid=True,
            goodness_of_fit=np.sum(goodness_of_fit),
        )

        return shower_result, energy_result

    def minimise(
        self,
        params,
        step,
        limits,
    ):
        """

        Parameters
        ----------
        params: ndarray
            Seed parameters for fit
        step: ndarray
            Initial step size in the fit
        limits: ndarray
            Fit bounds
        Returns
        -------
        tuple: best fit parameters and errors
        """
        limits = np.asarray(limits)

        energy = params[4]
        xmax_scale = 1

        # Now do the minimisation proper
        minimizer = Minuit(
            self.get_likelihood,
            source_x=params[0],
            source_y=params[1],
            core_x=params[2],
            core_y=params[3],
            energy=energy,
            x_max_scale=xmax_scale,
            goodness_of_fit=False,
        )
        # This time leave everything free
        minimizer.fixed = [False, False, False, False, False, False, True]
        minimizer.errors = step
        minimizer.limits = limits
        minimizer.errordef = MINUIT_ERRORDEF

        # Tighter fit tolerances
        minimizer.tol *= MINUIT_TOLERANCE_FACTOR
        minimizer.strategy = MINUIT_STRATEGY

        # Fit and output parameters and errors
        _ = minimizer.migrad(iterate=MIGRAD_ITERATE)
        fit_params = minimizer.values
        errors = minimizer.errors

        return (
            (
                fit_params["source_x"],
                fit_params["source_y"],
                fit_params["core_x"],
                fit_params["core_y"],
                fit_params["energy"],
                fit_params["x_max_scale"],
            ),
            (
                errors["source_x"],
                errors["source_y"],
                errors["core_x"],
                errors["core_x"],
                errors["energy"],
                errors["x_max_scale"],
            ),
            minimizer.fval,
        )

    def reset_interpolator(self):
        """
        This function is needed in order to reset some variables in the interpolator
        at each new event. Without this reset, a new event starts with information
        from the previous event.
        """
        for key in self.prediction:
            self.prediction[key].reset()
        for key in self.time_prediction:
            self.time_prediction[key].reset()
