#!/usr/bin/env python3
"""

"""
import math

import numpy as np
from astropy import units as u
from iminuit import Minuit

from ctapipe.coordinates import (HorizonFrame,
                                 NominalFrame,
                                 TiltedGroundFrame,
                                 GroundFrame,
                                 project_to_ground)
from ctapipe.image import poisson_likelihood_gaussian
from ctapipe.io.containers import (ReconstructedShowerContainer,
                                   ReconstructedEnergyContainer)
from ctapipe.reco.reco_algorithms import Reconstructor
from ctapipe.utils import TableInterpolator
from ctapipe.instrument import get_atmosphere_profile_functions

from scipy.optimize import minimize, least_squares
from scipy.stats import norm

__all__ = ['ImPACTReconstructor', 'energy_prior', 'xmax_prior']


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
    x_max_exp = 300 * (u.g * u.cm**-2) + \
        93 * (u.g * u.cm**-2) * np.log10(energy.to(u.TeV).value)

    return x_max_exp


def energy_prior(energy, index=-1):

    return -2 * np.log(np.power(energy, index))


def xmax_prior(energy, xmax, width=30):

    x_max_exp = guess_shower_depth(energy)
    diff = xmax.value - x_max_exp

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

    def __init__(self, root_dir=".", minimiser="minuit", prior=""):

        # First we create a dictionary of image template interpolators
        # for each telescope type
        self.root_dir = root_dir
        self.prediction = dict()

        self.file_names = {"CHEC": "GCT_xm_full.fits", "LSTCam": "LST_xm_full.fits",
                           "NectarCam": "MST_xm_full.fits",
                           "FlashCam": "MST_xm_full.fits"}

        # We also need a conversion function from height above ground to
        # depth of maximum To do this we need the conversion table from CORSIKA
        self.thickness_profile, self.altitude_profile = \
            get_atmosphere_profile_functions('paranal')

        # For likelihood calculation we need the with of the
        # pedestal distribution for each pixel
        # currently this is not availible from the calibration,
        # so for now lets hard code it in a dict
        self.ped_table = {"LSTCam": 1.3,
                          "NectarCam": 2.0,
                          "FlashCam": 2.3,
                          "CHEC": 1.3}
        self.spe = 0.5  # Also hard code single p.e. distribution width

        # Also we need to scale the impact_reco templates a bit, this will be
        #  fixed later
        self.scale = {"LSTCam": 1.3, "NectarCam": 1.1, "FlashCam": 1.4, "CHEC": 1.0}  # * 1.36}

        self.last_image = dict()
        self.last_point = dict()

        # Next we need the position, area and amplitude from each pixel in the event
        # making this a class member makes passing them around much easier

        self.pixel_x = 0
        self.pixel_y = 0
        self.pixel_area = 0
        self.image = 0
        self.type = ("LST")
        # We also need telescope positions
        self.tel_pos_x = 0
        self.tel_pos_y = 0
        # And the peak of the images
        self.peak_x = 0
        self.peak_y = 0
        self.peak_amp = 0
        self.hillas = 0

        self.ped = dict()

        self.array_direction = 0
        self.minimiser_name = minimiser

        self.array_return = False
        self.priors = prior

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
            if tel_type[t] in self.prediction.keys():
                continue

            self.prediction[tel_type[t]] = \
                TableInterpolator(self.root_dir + "/" +
                                  self.file_names[tel_type[t]])

        return True

    def get_brightest_mean(self):
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
        peak_x = np.zeros(
            [len(self.pixel_x)])  # Create blank arrays for peaks
        # rather than a dict (faster)
        peak_y = np.zeros(peak_x.shape)
        peak_amp = np.zeros(peak_x.shape)

        # Loop over all tels to take weighted average of pixel
        # positions This loop could maybe be replaced by an array
        # operation by a numpy wizard

        tel_num = 0
        for tel in self.hillas:

            weight = self.hillas[tel].size
            weighted_x = self.hillas[tel].cen_x.to(u.rad).value * weight
            weighted_y = self.hillas[tel].cen_y.to(u.rad).value * weight

            ppx = np.sum(weighted_x) / np.sum(weight)
            ppy = np.sum(weighted_y) / np.sum(weight)

            peak_x[tel_num] = ppx  # Fill up array
            peak_y[tel_num] = ppy
            peak_amp[tel_num] = np.sum(weight)
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
        impact = np.sqrt(np.power(np.array(list(self.tel_pos_x.values()))
                                  - core_x, 2) +
                         np.power(np.array(list(self.tel_pos_y.values()))
                                  - core_y, 2))

        # Distance above telescope is ratio of these two (small angle)

        height = impact / disp
        weight = np.power(self.peak_amp, 0.)  # weight average by amplitude
        hm = height * u.m
        hm[hm > 99 * u.km] = 99 * u.km

        # Take weighted mean of esimates
        mean_height = np.sum(height * weight) / np.sum(weight)
        # This value is height above telescope in the tilted system,
        # we should convert to height above ground
        mean_height *= np.cos(zen)

        # Add on the height of the detector above sea level
        mean_height += 2100

        if mean_height > 100000 or np.isnan(mean_height):
            mean_height = 100000

        mean_height *= u.m
        # Lookup this height in the depth tables, the convert Hmax to Xmax
        x_max = self.thickness_profile(mean_height.to(u.km))
        # self.shower_max.interpolate(mean_height.to(u.km))

        # Convert to slant depth
        x_max /= np.cos(zen)
        return x_max

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

        pixel_pos_trans_x = (pixel_pos_x - x_trans) * \
            np.cos(phi) - (pixel_pos_y - y_trans) * np.sin(phi)
        pixel_pos_trans_y = (pixel_pos_x - x_trans) * \
            np.sin(phi) + (pixel_pos_y - y_trans) * np.cos(phi)
        return pixel_pos_trans_x, pixel_pos_trans_y

    def image_prediction(self, type, energy, impact, x_max, pix_x, pix_y):
        """Creates predicted image for the specified pixels, interpolated
        from the template library.

        Parameters
        ----------
        type: string
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

        return self.prediction[type].interpolate([energy, impact,
                                                  x_max], pix_x, pix_y)

    def get_prediction(self, tel_id, shower_reco, energy_reco):

        horizon_seed = HorizonFrame(az=shower_reco.az, alt=shower_reco.alt)
        nominal_seed = horizon_seed.transform_to(
            NominalFrame(array_direction=horizon_seed))
        source_x = nominal_seed.x.to(u.rad).value
        source_y = nominal_seed.y.to(u.rad).value

        ground = GroundFrame(x=shower_reco.core_x, y=shower_reco.core_y, z=0 * u.m)
        tilted = ground.transform_to(
            TiltedGroundFrame(pointing_direction=self.array_direction))
        tilt_x = tilted.x.to(u.m).value
        tilt_y = tilted.y.to(u.m).value

        zenith = 90 * u.deg - self.array_direction.alt

        x_max = shower_reco.h_max / np.cos(zenith)

        # Calculate expected Xmax given this energy
        x_max_exp = guess_shower_depth(energy_reco.energy)

        # Convert to binning of Xmax, addition of 100 can probably be removed
        x_max_bin = x_max - x_max_exp

        # Check for range
        if x_max_bin > 250 * (u.g * u.cm**-2):
            x_max_bin = 250 * (u.g * u.cm**-2)
        if x_max_bin < -250 * (u.g * u.cm**-2):
            x_max_bin = -250 * (u.g * u.cm**-2)

        x_max_bin = x_max_bin.value

        impact = np.sqrt(pow(self.tel_pos_x[tel_id] - tilt_x, 2) +
                         pow(self.tel_pos_y[tel_id] - tilt_y, 2))

        phi = np.arctan2((self.tel_pos_y[tel_id] - tilt_y),
                         (self.tel_pos_x[tel_id] - tilt_x))

        pix_x_rot, pix_y_rot = self.rotate_translate(self.pixel_x[tel_id] * -1,
                                                     self.pixel_y[tel_id],
                                                     source_x,
                                                     source_y, phi)

        prediction = self.image_prediction(self.type[tel_id],
                                           # (90 * u.deg) - shower_reco.alt,
                                           # shower_reco.az,
                                           energy_reco.energy.value,
                                           impact, x_max_bin,
                                           pix_x_rot * (180 / math.pi),
                                           pix_y_rot * (180 / math.pi))

        prediction *= self.scale[self.type[tel_id]]
        # prediction *= self.pixel_area[tel_id]

        prediction[prediction < 0] = 0
        prediction[np.isnan(prediction)] = 0

        return prediction

    def get_likelihood(self, source_x, source_y, core_x, core_y,
                       energy, x_max_scale):
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

        # First we add units back onto everything.  Currently not
        # handled very well, maybe in future we could just put
        # everything in the correct units when loading in the class
        # and ignore them from then on

        zenith = 90 * u.deg - self.array_direction.alt
        azimuth = self.array_direction.az

        # Geometrically calculate the depth of maximum given this test position
        x_max = self.get_shower_max(source_x, source_y,
                                    core_x, core_y,
                                    zenith.to(u.rad).value) * x_max_scale
        # Calculate expected Xmax given this energy
        x_max_exp = guess_shower_depth(energy * u.TeV)

        # Convert to binning of Xmax, addition of 100 can probably be removed
        x_max_bin = x_max - x_max_exp

        # Check for range
        if x_max_bin > 250 * (u.g * u.cm**-2):
            x_max_bin = 250 * (u.g * u.cm**-2)
        if x_max_bin < -250 * (u.g * u.cm**-2):
            x_max_bin = -250 * (u.g * u.cm**-2)

        x_max_bin = x_max_bin.value

        array_like = None

        for tel_count in self.image:  # Loop over all telescopes
            # Calculate impact distance for all telescopes
            impact = np.sqrt(pow(self.tel_pos_x[tel_count] - core_x, 2)
                             + pow(self.tel_pos_y[tel_count] - core_y, 2))
            # And the expected rotation angle
            phi = np.arctan2((self.tel_pos_y[tel_count] - core_y),
                             (self.tel_pos_x[tel_count] - core_x))  # - (math.pi/2.)

            # Rotate and translate all pixels such that they match the
            # template orientation
            pix_x_rot, pix_y_rot = self.rotate_translate(
                self.pixel_x[tel_count] * -1,
                self.pixel_y[tel_count],
                source_x, source_y, phi
            )

            # Then get the predicted image, convert pixel positions to deg
            prediction = self.image_prediction(
                self.type[tel_count],
                #zenith, azimuth,
                energy, impact, x_max_bin,
                pix_x_rot * (180 / math.pi),
                pix_y_rot * (180 / math.pi)
            )
            prediction[np.isnan(prediction)] = 0
            prediction[prediction < 1e-6] = 1e-6

            # Scale templates to match simulations
            prediction *= self.scale[self.type[tel_count]]
            # prediction *= self.pixel_area[tel_count]

            # Get likelihood that the prediction matched the camera image
            like = poisson_likelihood_gaussian(self.image[tel_count],
                                               prediction,
                                               self.spe,
                                               self.ped[tel_count])
            if np.any(prediction == np.inf):
                print("inf found at ", self.type[tel_count], zenith,
                      azimuth, energy, impact, x_max_bin)
            like[np.isnan(like)] = 1e9
            if array_like is None:
                array_like = like
            else:
                array_like = np.append(array_like, like)

        prior_pen = 0
        # Add prior penalities if we have them
        array_like += 1e-8
        if "energy" in self.priors:
            prior_pen += energy_prior(energy, index=-2)
        if "xmax" in self.priors:
            prior_pen += xmax_prior(energy, x_max)

        array_like += prior_pen / float(len(array_like))
        if self.array_return:
            return array_like
        return np.sum(array_like)

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
        return self.get_likelihood(x[0], x[1], x[2], x[3], x[4], x[5])

    def set_event_properties(self, image, pixel_x, pixel_y,
                             pixel_area, type_tel, tel_x, tel_y,
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

        self.pixel_x = dict()
        self.pixel_y = dict()

        self.tel_pos_x = dict()
        self.tel_pos_y = dict()
        self.pixel_area = dict()
        self.ped = dict()

        # So here we must loop over the telescopes
        for x in tel_x:
            self.pixel_x[x] = pixel_x[x].to(u.rad).value
            self.pixel_y[x] = pixel_y[x].to(u.rad).value

            self.tel_pos_x[x] = tel_x[x].value
            self.tel_pos_y[x] = tel_y[x].value
            self.pixel_area[x] = pixel_area[x].to(u.deg * u.deg).value
            # Here look up pedestal value
            self.ped[x] = self.ped_table[type_tel[x]]

        self.hillas = hillas

        self.get_brightest_mean()
        self.type = type_tel
        self.initialise_templates(type_tel)

        self.array_direction = array_direction
        self.last_image = 0
        self.last_point = 0

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

        horizon_seed = HorizonFrame(az=shower_seed.az, alt=shower_seed.alt)
        nominal_seed = horizon_seed.transform_to(NominalFrame(
            array_direction=self.array_direction))

        source_x = nominal_seed.x[0].to(u.rad).value
        source_y = nominal_seed.y[0].to(u.rad).value

        ground = GroundFrame(x=shower_seed.core_x,
                             y=shower_seed.core_y, z=0 * u.m)
        tilted = ground.transform_to(
            TiltedGroundFrame(pointing_direction=self.array_direction)
        )
        tilt_x = tilted.x.to(u.m).value
        tilt_y = tilted.y.to(u.m).value

        lower_en_limit = energy_seed.energy * 0.5
        en_seed = energy_seed.energy
        if lower_en_limit < 0.04 * u.TeV:
            lower_en_limit = 0.04 * u.TeV
            en_seed = 0.041 * u.TeV

        seed = (source_x, source_y, tilt_x,
                tilt_y, en_seed.value, 0.8)
        step = (0.001, 0.001, 10, 10, en_seed.value * 0.1, 0.1)
        limits = ((source_x - 0.01, source_x + 0.01),
                  (source_y - 0.01, source_y + 0.01),
                  (tilt_x - 100, tilt_x + 100),
                  (tilt_y - 100, tilt_y + 100),
                  (lower_en_limit.value, en_seed.value * 2),
                  (0.5, 2))

        fit_params, errors = self.minimise(params=seed, step=step, limits=limits,
                                           minimiser_name=self.minimiser_name)

        # container class for reconstructed showers '''
        shower_result = ReconstructedShowerContainer()

        nominal = NominalFrame(x=fit_params[0] * u.rad,
                               y=fit_params[1] * u.rad,
                               array_direction=self.array_direction)
        horizon = nominal.transform_to(HorizonFrame())

        shower_result.alt, shower_result.az = horizon.alt, horizon.az
        tilted = TiltedGroundFrame(x=fit_params[2] * u.m,
                                   y=fit_params[3] * u.m,
                                   pointing_direction=self.array_direction)
        ground = project_to_ground(tilted)

        shower_result.core_x = ground.x
        shower_result.core_y = ground.y

        shower_result.is_valid = True

        shower_result.alt_uncert = np.nan
        shower_result.az_uncert = np.nan
        shower_result.core_uncert = np.nan

        zenith = 90 * u.deg - self.array_direction.alt
        shower_result.h_max = fit_params[5] * \
            self.get_shower_max(fit_params[0],
                                fit_params[1],
                                fit_params[2],
                                fit_params[3],
                                zenith.to(u.rad).value)

        shower_result.h_max_uncert = errors[5] * shower_result.h_max

        shower_result.goodness_of_fit = np.nan
        shower_result.tel_ids = list(self.image.keys())

        energy_result = ReconstructedEnergyContainer()
        energy_result.energy = fit_params[4] * u.TeV
        energy_result.energy_uncert = errors[4] * u.TeV
        energy_result.is_valid = True
        energy_result.tel_ids = list(self.image.keys())
        # Return interesting stuff

        return shower_result, energy_result

    def minimise(self, params, step, limits, minimiser_name="minuit"):
        """

        Parameters
        ----------
        params
        step
        limits
        minimiser_name

        Returns
        -------

        """
        if minimiser_name == "minuit":

            min = Minuit(self.get_likelihood,
                         print_level=1,
                         source_x=params[0],
                         error_source_x=step[0],
                         limit_source_x=limits[0],
                         source_y=params[1],
                         error_source_y=step[1],
                         limit_source_y=limits[1],
                         core_x=params[2],
                         error_core_x=step[2],
                         limit_core_x=limits[2],
                         core_y=params[3],
                         error_core_y=step[3],
                         limit_core_y=limits[3],
                         energy=params[4],
                         error_energy=step[4],
                         limit_energy=limits[4],
                         x_max_scale=params[5], error_x_max_scale=step[5],
                         limit_x_max_scale=limits[5],
                         fix_x_max_scale=False,
                         errordef=1)

            min.migrad()

            min.tol *= 1000
            min.set_strategy(0)

            # Perform minimisation
            fit_params = min.values
            errors = min.errors

            return (fit_params["source_x"], fit_params["source_y"], fit_params["core_x"],
                    fit_params["core_y"], fit_params["energy"], fit_params[
                        "x_max_scale"]),\
                (errors["source_x"], errors["source_y"], errors["core_x"],
                 errors["core_x"], errors["energy"], errors["x_max_scale"])

        elif minimiser_name in ("lm", "trf", "dogleg"):
            self.array_return = True

            min = least_squares(self.get_likelihood_min, params,
                                method=minimiser_name,
                                x_scale=step,
                                xtol=1e-10,
                                ftol=1e-10
                                )
            return min.x, (0, 0, 0, 0, 0, 0)

        else:
            min = minimize(self.get_likelihood_min, params,
                           method=minimiser_name,
                           bounds=limits
                           )
            return min.x, (0, 0, 0, 0, 0, 0)

    def draw_nominal_surface(self, shower_seed, energy_seed, bins=30,
                             nominal_range=2.5 * u.deg):
        """
        Simple reconstruction for evaluating the likelihood in a grid across the 
        nominal system, fixing all values but the source position of the gamma rays. 
        Useful for checking the reconstruction performance of the algorithm

        Parameters
        ----------
        shower_seed: ReconstructedShowerContainer
            Best fit ImPACT shower geometry 
        energy_seed: ReconstructedEnergyContainer
            Best fit ImPACT energy
        bins: int
            Number of bins in surface evaluation
        nominal_range: Quantity
            Range over which to create likelihood surface

        Returns
        -------
        ndarray, ndarray, ndarray: 
        Bin centres in X and Y coordinates and the values of the likelihood at each 
        position
        """
        horizon_seed = HorizonFrame(az=shower_seed.az, alt=shower_seed.alt)
        nominal_seed = horizon_seed.transform_to(
            NominalFrame(array_direction=self.array_direction))

        source_x = nominal_seed.x[0].to(u.rad)
        source_y = nominal_seed.y[0].to(u.rad)

        ground = GroundFrame(x=shower_seed.core_x,
                             y=shower_seed.core_y, z=0 * u.m)
        tilted = ground.transform_to(
            TiltedGroundFrame(pointing_direction=self.array_direction)
        )
        tilt_x = tilted.x.to(u.m)
        tilt_y = tilted.y.to(u.m)

        x_dir = np.linspace(source_x - nominal_range,
                            source_x + nominal_range, num=bins)
        y_dir = np.linspace(source_y - nominal_range,
                            source_y + nominal_range, num=bins)
        w = np.zeros([bins, bins])
        zenith = 90 * u.deg - self.array_direction.alt

        for xb in range(bins):
            for yb in range(bins):
                shower_max = self.get_shower_max(x_dir[xb].to(u.rad).value,
                                                 y_dir[yb].to(u.rad).value,
                                                 tilt_x.value,
                                                 tilt_y.value,
                                                 zenith.to(u.rad).value)
                x_max_scale = shower_seed.h_max / shower_max

                w[xb][yb] = self.get_likelihood(x_dir[xb].to(u.rad).value,
                                                y_dir[yb].to(u.rad).value,
                                                tilt_x.value,
                                                tilt_y.value,
                                                energy_seed.energy.value,
                                                x_max_scale)

        w = w - np.min(w)

        return x_dir.to(u.deg), y_dir.to(u.deg), w

    def draw_tilted_surface(self, shower_seed, energy_seed,
                            bins=50, core_range=100 * u.m):
        """
        Simple reconstruction for evaluating the likelihood in a grid across
        the nominal system, fixing all values but the core position of the
        gamma rays. Useful for checking the reconstruction performance of the
        algorithm

        Parameters
        ----------
        shower_seed: ReconstructedShowerContainer
            Best fit ImPACT shower geometry 
        energy_seed: ReconstructedEnergyContainer
            Best fit ImPACT energy
        bins: int
            Number of bins in surface evaluation
        nominal_range: Quantity
            Range over which to create likelihood surface

        Returns
        -------
        ndarray, ndarray, ndarray: 
            Bin centres in X and Y coordinates and the values of the likelihood
            at each position
        """
        horizon_seed = HorizonFrame(az=shower_seed.az, alt=shower_seed.alt)
        nominal_seed = horizon_seed.transform_to(
            NominalFrame(array_direction=self.array_direction)
        )

        source_x = nominal_seed.x[0].to(u.rad).value
        source_y = nominal_seed.y[0].to(u.rad).value

        ground = GroundFrame(
            x=shower_seed.core_x,
            y=shower_seed.core_y, z=0 * u.m
        )
        tilted = ground.transform_to(
            TiltedGroundFrame(pointing_direction=self.array_direction)
        )
        tilt_x = tilted.x.to(u.m)
        tilt_y = tilted.y.to(u.m)

        x_ground_list = np.linspace(tilt_x - core_range,
                                    tilt_x + core_range, num=bins)
        y_ground_list = np.linspace(tilt_y - core_range,
                                    tilt_y + core_range, num=bins)
        w = np.zeros([bins, bins])
        zenith = 90 * u.deg - self.array_direction.alt

        for xb in range(bins):
            for yb in range(bins):
                shower_max = self.get_shower_max(source_x,
                                                 source_y,
                                                 x_ground_list[xb].value,
                                                 y_ground_list[yb].value,
                                                 zenith.to(u.rad).value)
                x_max_scale = shower_seed.h_max / shower_max

                w[xb][yb] = self.get_likelihood(source_x,
                                                source_y,
                                                x_ground_list[xb].value,
                                                y_ground_list[yb].value,
                                                energy_seed.energy.value,
                                                x_max_scale)
        return x_ground_list, y_ground_list, w
