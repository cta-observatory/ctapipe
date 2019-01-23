#!/usr/bin/env python3
"""

"""
import math

import numpy as np
import numpy.ma as ma
from astropy import units as u
from astropy.coordinates import SkyCoord
from iminuit import Minuit
from scipy.optimize import minimize, least_squares
from scipy.stats import norm

from ctapipe.coordinates import (HorizonFrame,
                                 NominalFrame,
                                 TiltedGroundFrame,
                                 GroundFrame,
                                 project_to_ground)
from ctapipe.image import poisson_likelihood_gaussian, mean_poisson_likelihood_gaussian
from ctapipe.instrument import get_atmosphere_profile_functions
from ctapipe.io.containers import (ReconstructedShowerContainer,
                                   ReconstructedEnergyContainer)
from ctapipe.reco.reco_algorithms import Reconstructor
from ctapipe.utils.template_network_interpolator import TemplateNetworkInterpolator, \
    TimeGradientInterpolator

__all__ = ['ImPACTReconstructor', 'energy_prior', 'xmax_prior', 'guess_shower_depth']


def guess_shower_depth(energy):
    """
    Simple estimation of depth of shower max based on the expected gamma-ray elongation
    rate.

    Parameters
    ----------
    energy: float
        Energy of the shower in TeV

    Returns
    -------
    float: Expected depth of shower maximum
    """

    x_max_exp = 300 + 93 * np.log10(energy)

    return x_max_exp


def energy_prior(energy, index=-1):
    return -2 * np.log(np.power(energy, index))


def xmax_prior(energy, xmax, width=100):
    x_max_exp = guess_shower_depth(energy)
    diff = xmax - x_max_exp
    return -2 * np.log(norm.pdf(diff / width))


class ImPACTReconstructor(Reconstructor):
    """This class is an implementation if the impact_reco Monte Carlo
    Template based image fitting method from parsons14.  This method uses a
    comparision of the predicted image from a library of image
    templates to perform a maximum likelihood fit for the shower axis,
    energy and height of maximum.

    Because this application is computationally intensive the usual
    advice to use astropy units for all quantities is ignored (as
    these slow down some computations), instead units within the class
    are fixed:

    - Angular units in radians
    - Distance units in metres
    - Energy units in TeV

    References
    ----------
    .. [parsons14] Parsons & Hinton, Astroparticle Physics 56 (2014), pp. 26-34

    """

    # For likelihood calculation we need the with of the
    # pedestal distribution for each pixel
    # currently this is not availible from the calibration,
    # so for now lets hard code it in a dict
    ped_table = {"LSTCam": 2.8,
                 "NectarCam": 2.3,
                 "FlashCam": 2.3,
                 "CHEC": 0.5,
                 "DUMMY": 0}
    spe = 0.5  # Also hard code single p.e. distribution width

    def __init__(self, root_dir=".", minimiser="minuit", prior="",
                 template_scale=1., xmax_offset=0, use_time_gradient=False):

        # First we create a dictionary of image template interpolators
        # for each telescope type
        self.root_dir = root_dir
        self.priors = prior
        self.minimiser_name = minimiser

        self.file_names = {"CHEC": ["GCT_05deg_ada.template.gz",
                                    "GCT_05deg_time.template.gz"],
                           "LSTCam": ["LST_05deg.template.gz",
                                      "LST_05deg_time.template.gz"],
                           "NectarCam": ["MST_05deg.template.gz",
                                         "MST_05deg_time.template.gz"],
                           "FlashCam": ["MST_xm_full.fits"]}

        # We also need a conversion function from height above ground to
        # depth of maximum To do this we need the conversion table from CORSIKA
        self.thickness_profile, self.altitude_profile = \
            get_atmosphere_profile_functions('paranal', with_units=False)

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
        self.array_return = False

        # For now these factors are required to fix problems in templates
        self.template_scale = template_scale
        self.xmax_offset = xmax_offset
        self.use_time_gradient = use_time_gradient

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

            self.prediction[tel_type[t]] = \
                TemplateNetworkInterpolator(self.root_dir + "/" +
                                            self.file_names[tel_type[t]][0])
            if self.use_time_gradient:
                self.time_prediction[tel_type[t]] = \
                    TimeGradientInterpolator(self.root_dir + "/" +
                                             self.file_names[tel_type[t]][1])

        return True

    def get_hillas_mean(self):
        """This is a simple function to find the peak position of each image
        in an event which will be used later in the Xmax calculation. Peak is
        found by taking the average position of the n hottest pixels in the
        image.

        Parameters
        ----------

        Returns
        -------
            None

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
            peak_x[tel_num] = hillas.x.to(u.rad).value   # Fill up array
            peak_y[tel_num] = hillas.y.to(u.rad).value
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
        disp = np.sqrt(np.power(self.peak_x - source_x, 2) +
                       np.power(self.peak_y - source_y, 2))
        # Calculate impact parameter of the shower
        impact = np.sqrt(np.power(self.tel_pos_x - core_x, 2) +
                         np.power(self.tel_pos_y - core_y, 2))
        # Distance above telescope is ratio of these two (small angle)

        height = impact / disp
        weight = np.power(self.peak_amp, 0.)  # weight average by sqrt amplitude
        # sqrt may not be the best option...

        # Take weighted mean of estimates
        mean_height = np.sum(height * weight) / np.sum(weight)
        # This value is height above telescope in the tilted system,
        # we should convert to height above ground
        mean_height *= np.cos(zen)

        # Add on the height of the detector above sea level
        mean_height += 2150

        if mean_height > 100000 or np.isnan(mean_height):
            mean_height = 100000

        # Lookup this height in the depth tables, the convert Hmax to Xmax
        x_max = self.thickness_profile(mean_height)
        # Convert to slant depth
        x_max /= np.cos(zen)

        return x_max + self.xmax_offset

    @staticmethod
    def rotate_translate(pixel_pos_x, pixel_pos_y, x_trans, y_trans, phi):
        """
        Function to perform rotation and translation of pixel lists

        Parameters
        ----------
        pixel_pos_x: ndarray
            Array of pixel x positions
        pixel_pos_y: ndarray
            Array of pixel x positions
        x_trans: float
            Translation of position in x coordinates
        y_trans: float
            Translation of position in y coordinates
        phi: float
            Rotation angle of pixels

        Returns
        -------
            ndarray,ndarray: Transformed pixel x and y coordinates

        """

        cosine_angle = np.cos(phi[..., np.newaxis])
        sin_angle = np.sin(phi[..., np.newaxis])

        pixel_pos_trans_x = (x_trans - pixel_pos_x ) * cosine_angle - \
                            (y_trans - pixel_pos_y ) * sin_angle

        pixel_pos_trans_y = (pixel_pos_x - x_trans) * sin_angle + \
                            (pixel_pos_y - y_trans) * cosine_angle
        return pixel_pos_trans_x, pixel_pos_trans_y

    def image_prediction(self, tel_type, energy, impact, x_max, pix_x, pix_y):
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

        return self.prediction[tel_type](energy, impact, x_max, pix_x, pix_y)

    def predict_time(self, tel_type, energy, impact, x_max):
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
        return self.time_prediction[tel_type](energy, impact, x_max)

    def get_likelihood(self, source_x, source_y, core_x, core_y,
                       energy, x_max_scale, goodness_of_fit=False):
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
        # First we add units back onto everything.  Currently not
        # handled very well, maybe in future we could just put
        # everything in the correct units when loading in the class
        # and ignore them from then on

        zenith = (np.pi / 2) - self.array_direction.alt.to(u.rad).value
        azimuth = self.array_direction.az

        # Geometrically calculate the depth of maximum given this test position
        x_max = self.get_shower_max(source_x, source_y,
                                    core_x, core_y,
                                    zenith)
        x_max *= x_max_scale

        # Calculate expected Xmax given this energy
        x_max_exp = guess_shower_depth(energy)  # / np.cos(20*u.deg)

        # Convert to binning of Xmax
        x_max_bin = x_max - x_max_exp

        # Check for range
        if x_max_bin > 200:
            x_max_bin = 200
        if x_max_bin < -100:
            x_max_bin = -100

        # Calculate impact distance for all telescopes
        impact = np.sqrt(np.power(self.tel_pos_x - core_x, 2)
                         + np.power(self.tel_pos_y - core_y, 2))
        # And the expected rotation angle
        phi = np.arctan2((self.tel_pos_x - core_x),
                         (self.tel_pos_y - core_y)) * u.rad

        # Rotate and translate all pixels such that they match the
        # template orientation
        pix_y_rot, pix_x_rot = self.rotate_translate(
            self.pixel_x,
            self.pixel_y,
            source_x, source_y, phi
        )

        # In the interpolator class we can gain speed advantages by using masked arrays
        # so we need to make sure here everything is masked
        prediction = ma.zeros(self.image.shape)
        prediction.mask = ma.getmask(self.image)

        time_gradients = np.zeros((self.image.shape[0],2))

        # Loop over all telescope types and get prediction
        for tel_type in np.unique(self.tel_types).tolist():
            type_mask = self.tel_types == tel_type
            prediction[type_mask] = \
                self.image_prediction(tel_type, energy *
                                      np.ones_like(impact[type_mask]),
                                      impact[type_mask], x_max_bin *
                                      np.ones_like(impact[type_mask]),
                                      pix_x_rot[type_mask] * (180 / math.pi) * -1,
                                      pix_y_rot[type_mask] * (180 / math.pi))

            if self.use_time_gradient:
                time_gradients[type_mask] = \
                    self.predict_time(tel_type,
                                      energy * np.ones_like(impact[type_mask]),
                                      impact[type_mask],
                                      x_max_bin * np.ones_like(impact[type_mask]))

        if self.use_time_gradient:
            time_mask = np.logical_and(np.invert(ma.getmask(self.image)),
                                       self.time > 0)
            weight = np.sqrt(self.image) * time_mask
            rv = norm()

            sx = pix_x_rot * weight
            sxx = pix_x_rot * pix_x_rot * weight

            sy = self.time * weight
            sxy = self.time * pix_x_rot * weight
            d = weight.sum(axis=1) * sxx.sum(axis=1) - sx.sum(axis=1) * sx.sum(axis=1)
            time_fit = (weight.sum(axis=1) * sxy.sum(axis=1) - sx.sum(axis=1) * sy.sum(
                axis=1)) / d
            time_fit /= -1 * (180 / math.pi)
            chi2 = -2 * np.log(rv.pdf((time_fit - time_gradients.T[0])/
                                        time_gradients.T[1]))

        # Likelihood function will break if we find a NaN or a 0
        prediction[np.isnan(prediction)] = 1e-8
        prediction[prediction < 1e-8] = 1e-8
        prediction *= self.template_scale

        # Get likelihood that the prediction matched the camera image
        like = poisson_likelihood_gaussian(self.image, prediction, self.spe, self.ped)
        like[np.isnan(like)] = 1e9
        like *= np.invert(ma.getmask(self.image))
        like = ma.MaskedArray(like, mask=ma.getmask(self.image))

        array_like = like
        if goodness_of_fit:
            return np.sum(like - mean_poisson_likelihood_gaussian(prediction, self.spe,
                                                                  self.ped))

        prior_pen = 0
        # Add prior penalities if we have them
        array_like += 1e-8
        if "energy" in self.priors:
            prior_pen += energy_prior(energy, index=-1)
        if "xmax" in self.priors:
            prior_pen += xmax_prior(energy, x_max)

        array_like += prior_pen / float(len(array_like))

        if self.array_return:
            array_like = array_like.ravel()
            return array_like[np.invert(ma.getmask(array_like))]

        final_sum = array_like.sum()
        if self.use_time_gradient:
            final_sum += chi2.sum() #* np.sum(ma.getmask(self.image))

        return final_sum

    def get_likelihood_min(self, x):
        """Wrapper class around likelihood function for use with scipy
        minimisers

        Parameters
        ----------
        x: ndarray
            Array of minimisation parameters

        Returns
        -------
        float: Likelihood value of test position

        """

        val = self.get_likelihood(x[0], x[1], x[2], x[3], x[4], x[5])

        return val

    def get_likelihood_nlopt(self, x, grad):
        """Wrapper class around likelihood function for use with scipy
        minimisers

        Parameters
        ----------
        x: ndarray
            Array of minimisation parameters

        Returns
        -------
        float: Likelihood value of test position

        """

        val = self.get_likelihood(x[0], x[1], x[2], x[3], x[4], x[5])
        return val

    def set_event_properties(self, image, time, pixel_x, pixel_y, type_tel, tel_x, tel_y,
                             array_direction, hillas):
        """The setter class is used to set the event properties within this
        class before minimisation can take place. This simply copies a
        bunch of useful properties to class members, so that we can
        use them later without passing all this information around.

        Parameters
        ----------
        image: dictionary
            Amplitude of pixels in camera images
        pixel_x: dictionary
            X position of pixels in nominal system
        pixel_y: dictionary
            Y position of pixels in nominal system
        pixel_area: dictionary
            Area of pixel in each telescope type
        type_tel: dictionary
            Type of telescope
        tel_x: dictionary
            X position of telescope
        tel_y: dictionary
            Y position of telescope

        Returns
        -------
        None

        """
        # First store these parameters in the class so we can use them
        # in minimisation For most values this is simply copying
        self.image = image

        self.tel_pos_x, self.tel_pos_y, self.ped = \
            np.zeros(len(tel_x)), np.zeros(len(tel_x)), np.zeros(len(tel_x))
        self.tel_types, self.tel_id = list(), list()

        max_pix_x, max_pix_y = 0, 0
        px, py, pa, pt = list(), list(), list(), list()
        self.hillas_parameters = list()

        # So here we must loop over the telescopes
        for x, i in zip(tel_x, range(len(tel_x))):

            px.append(pixel_x[x].to(u.rad).value)
            if len(px[i]) > max_pix_x:
                max_pix_x = len(px[i])
            py.append(pixel_y[x].to(u.rad).value)
            pa.append(image[x])
            pt.append(time[x])

            self.tel_pos_x[i] = tel_x[x].to(u.m).value
            self.tel_pos_y[i] = tel_y[x].to(u.m).value

            self.ped[i] = self.ped_table[type_tel[x]]
            self.tel_types.append(type_tel[x])
            self.tel_id.append(x)
            self.hillas_parameters.append(hillas[x])

        # Most interesting stuff is now copied to the class, but to remove our requirement
        # for loops we must copy the pixel positions to an array with the length of the
        # largest image

        # First allocate everything
        shape = (len(tel_x), max_pix_x)
        self.pixel_x, self.pixel_y = ma.zeros(shape), ma.zeros(shape)
        self.image, self.time, self.ped = ma.zeros(shape), ma.zeros(shape),\
                                          ma.zeros(shape)
        self.tel_types = np.array(self.tel_types)

        # Copy everything into our masked arrays
        for i in range(len(tel_x)):
            array_len = len(px[i])
            self.pixel_x[i][:array_len] = px[i]
            self.pixel_y[i][:array_len] = py[i]
            self.image[i][:array_len] = pa[i]
            self.time[i][:array_len] = pt[i]
            self.ped[i][:array_len] = self.ped_table[self.tel_types[i]]

        # Set the image mask
        mask = self.image == 0.0
        self.pixel_x[mask], self.pixel_y[mask] = ma.masked, ma.masked
        self.image[mask] = ma.masked
        self.time[mask] = ma.masked

        # Finally run some functions to get ready for the event
        self.get_hillas_mean()
        self.initialise_templates(type_tel)
        self.array_direction = array_direction

    def predict(self, shower_seed, energy_seed):
        """
        Parameters
        ----------
        shower_seed: ReconstructedShowerContainer
            Seed shower geometry to be used in the fit
        energy_seed: ReconstructedEnergyContainer
            Seed energy to be used in fit

        Returns
        -------
        ReconstructedShowerContainer, ReconstructedEnergyContainer:
        Reconstructed ImPACT shower geometry and energy
        """

        horizon_seed = SkyCoord(
            az=shower_seed.az, alt=shower_seed.alt, frame=HorizonFrame()
        )
        nominal_seed = horizon_seed.transform_to(
            NominalFrame(origin=self.array_direction)
        )

        source_x = nominal_seed.delta_az.to_value(u.rad)
        source_y = nominal_seed.delta_alt.to_value(u.rad)
        ground = GroundFrame(x=shower_seed.core_x,
                             y=shower_seed.core_y, z=0 * u.m)
        tilted = ground.transform_to(
            TiltedGroundFrame(pointing_direction=self.array_direction)
        )
        tilt_x = tilted.x.to(u.m).value
        tilt_y = tilted.y.to(u.m).value
        zenith = 90 * u.deg - self.array_direction.alt

        if len(self.hillas_parameters) > 3:
            shift = [1]
        else:
            shift = [1.5, 1, 0.5, 0, -0.5, -1, -1.5]

        seed_list = spread_line_seed(self.hillas_parameters,
                                     self.tel_pos_x, self.tel_pos_y,
                                     source_x[0], source_y[0], tilt_x, tilt_y,
                                     energy_seed.energy.value,
                                     shift_frac = shift)

        chosen_seed = self.choose_seed(seed_list)
        # Perform maximum likelihood fit
        fit_params, errors, like = self.minimise(params=chosen_seed[0],
                                                 step=chosen_seed[1],
                                                 limits=chosen_seed[2],
                                                 minimiser_name=self.minimiser_name)

        # Create a container class for reconstructed shower
        shower_result = ReconstructedShowerContainer()

        # Convert the best fits direction and core to Horizon and ground systems and
        # copy to the shower container
        nominal = SkyCoord(
            x=fit_params[0] * u.rad,
            y=fit_params[1] * u.rad,
            frame=NominalFrame(origin=self.array_direction)
        )
        horizon = nominal.transform_to(HorizonFrame())

        shower_result.alt, shower_result.az = horizon.alt, horizon.az
        tilted = TiltedGroundFrame(
            x=fit_params[2] * u.m,
            y=fit_params[3] * u.m,
            pointing_direction=self.array_direction
        )
        ground = project_to_ground(tilted)

        shower_result.core_x = ground.x
        shower_result.core_y = ground.y

        shower_result.is_valid = True

        # Currently no errors not availible to copy NaN
        shower_result.alt_uncert = np.nan
        shower_result.az_uncert = np.nan
        shower_result.core_uncert = np.nan

        # Copy reconstructed Xmax
        shower_result.h_max = fit_params[5] * self.get_shower_max(fit_params[0],
                                                                  fit_params[1],
                                                                  fit_params[2],
                                                                  fit_params[3],
                                                                  zenith.to(u.rad).value)

        shower_result.h_max *= np.cos(zenith)
        shower_result.h_max_uncert = errors[5] * shower_result.h_max

        shower_result.goodness_of_fit = like

        # Create a container class for reconstructed energy
        energy_result = ReconstructedEnergyContainer()
        # Fill with results
        energy_result.energy = fit_params[4] * u.TeV
        energy_result.energy_uncert = errors[4] * u.TeV
        energy_result.is_valid = True

        return shower_result, energy_result

    def choose_seed(self, seed_list):

        like = list()
        for seed in seed_list:
            #like.append(self.get_likelihood_min(seed[0]))
            like.append(self.minimise(seed[0], seed[1], seed[2],
                                      minimiser_name="nlopt",
                                      max_calls=10)[2])

        print("Choosing seed", np.argmin(like), like)
        return seed_list[np.argmin(like)]

    def minimise(self, params, step, limits, minimiser_name="minuit", max_calls=0):
        """

        Parameters
        ----------
        params: ndarray
            Seed parameters for fit
        step: ndarray
            Initial step size in the fit
        limits: ndarray
            Fit bounds
        minimiser_name: str
            Name of minimisation method
        max_calls: int
            Maximum number of calls to minimiser
        Returns
        -------
        tuple: best fit parameters and errors
        """
        limits = np.asarray(limits)
        if minimiser_name == "minuit":

            self.min = Minuit(self.get_likelihood,
                              print_level=1,
                              source_x=params[0], error_source_x=step[0],
                              limit_source_x=limits[0], fix_source_x=False,
                              source_y=params[1], error_source_y=step[1],
                              limit_source_y=limits[1], fix_source_y=False,
                              core_x=params[2], error_core_x=step[2],
                              limit_core_x=limits[2], fix_core_x=False,
                              core_y=params[3], error_core_y=step[3],
                              limit_core_y=limits[3], fix_core_y=False,
                              energy=params[4], error_energy=step[4],
                              limit_energy=limits[4], fix_energy=False,
                              x_max_scale=params[5], error_x_max_scale=step[5],
                              limit_x_max_scale=limits[5], fix_x_max_scale=False,
                              goodness_of_fit=False, fix_goodness_of_fit=True,
                              errordef=1)

            self.min.tol *= 1000
            self.min.set_strategy(1)

            migrad = self.min.migrad()
            fit_params = self.min.values
            errors = self.min.errors

            return (fit_params["source_x"], fit_params["source_y"], fit_params["core_x"],
                    fit_params["core_y"], fit_params["energy"], fit_params[
                        "x_max_scale"]), \
                   (errors["source_x"], errors["source_y"], errors["core_x"],
                    errors["core_x"], errors["energy"], errors["x_max_scale"]), \
                   self.min.fval

        elif "nlopt" in minimiser_name:
            import nlopt
            opt = nlopt.opt(nlopt.LN_BOBYQA, 6)
            opt.set_min_objective(self.get_likelihood_nlopt)
            opt.set_initial_step(step)

            opt.set_lower_bounds(np.asarray(limits).T[0])
            opt.set_upper_bounds(np.asarray(limits).T[1])
            opt.set_xtol_rel(1e-3)
            if max_calls:
                opt.set_maxeval(max_calls)

            x = opt.optimize(np.asarray(params))

            return x, (0, 0, 0, 0, 0, 0), self.get_likelihood_min(x)

        elif minimiser_name in ("lm", "trf", "dogleg"):
            self.array_return = True

            min = least_squares(self.get_likelihood_min, params,
                                method=minimiser_name,
                                x_scale=step,
                                xtol=1e-10,
                                ftol=1e-10,
                                )

            return min.x, (0, 0, 0, 0, 0, 0), self.get_likelihood_min(min.x)

        else:
            min = minimize(self.get_likelihood_min, np.array(params),
                           method=minimiser_name,
                           bounds=limits,
                           options={"disp": False},
                           tol=1e-5
                           )

            return np.array(min.x), (0, 0, 0, 0, 0, 0), self.get_likelihood_min(min.x)

def spread_line_seed(hillas, tel_x, tel_y, source_x, source_y, tilt_x, tilt_y, energy,
                     shift_frac = [2, 1.5, 1, 0.5, 0 ,-0.5, -1, -1.5]):
    """
    Parameters
    ----------
    hillas: list
        Hillas parameters in event
    tel_x: list
        telescope X positions in tilted system
    tel_y: list
        telescope Y positions in tilted system
    source_x: float
        Source X position in nominal system (radians)
    source_y:float
        Source Y position in nominal system (radians)
    tilt_x: float
        Core X position in tilited system (radians)
    tilt_y: float
        Core Y position in tilited system (radians)
    energy: float
        Energy in TeV
    shift_frac: list
        Fractional values to shist source and core positions

    Returns
    -------
    list of seed positions to try
    """
    centre_x, centre_y, amp = list(), list(), list()

    for tel_hillas in hillas:
        centre_x.append(tel_hillas.x.to(u.rad).value)
        centre_y.append(tel_hillas.y.to(u.rad).value)
        amp.append(tel_hillas.intensity)

    centre_x = np.average(centre_x, weights=amp)
    centre_y = np.average(centre_y, weights=amp)
    centre_tel_x = np.average(tel_x, weights=amp)
    centre_tel_y = np.average(tel_y, weights=amp)

    diff_x = source_x - centre_x
    diff_y = source_y - centre_y
    diff_tel_x = tilt_x - centre_tel_x
    diff_tel_y = tilt_y - centre_tel_y

    seed_list = list()

    for shift in shift_frac:
        seed_list.append(create_seed(centre_x + (diff_x*shift),
                                     centre_y + (diff_y*shift),
                                     centre_tel_x + (diff_tel_x * shift),
                                     centre_tel_y + (diff_tel_y * shift), energy))
    return seed_list


def create_seed(source_x, source_y, tilt_x, tilt_y, energy):
    """
    Function for creating seed, step and limits for a given position

    Parameters
    ----------
    source_x: float
        Source X position in nominal system (radians)
    source_y:float
        Source Y position in nominal system (radians)
    tilt_x: float
        Core X position in tilited system (radians)
    tilt_y: float
        Core Y position in tilited system (radians)
    energy: float
        Energy in TeV

    Returns
    -------
    tuple of seed, steps size and fit limits
    """
    lower_en_limit = energy * 0.5
    en_seed = energy

    # If our energy estimate falls outside of the range of our templates set it to
    # the edge
    if lower_en_limit < 0.01:
        lower_en_limit = 0.01
        en_seed = 0.01

    # Take the seed from Hillas-based reconstruction
    seed = (source_x, source_y, tilt_x,
            tilt_y, en_seed, 1)

    # Take a reasonable first guess at step size
    step = [0.04 / 57.3, 0.04 / 57.3, 5, 5, en_seed * 0.1, 0.05]
    # And some sensible limits of the fit range
    limits = [[source_x - 0.1, source_x + 0.1],
              [source_y - 0.1, source_y + 0.1],
              [tilt_x - 100, tilt_x + 100],
              [tilt_y - 100, tilt_y + 100],
              [lower_en_limit, en_seed * 2],
              [0.5, 2]
              ]

    return seed, step, limits
