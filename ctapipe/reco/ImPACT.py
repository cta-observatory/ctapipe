#!/usr/bin/env python3
"""

"""
import math
import numpy as np
from iminuit import Minuit
from astropy import units as u
from ctapipe.reco.table_interpolator import TableInterpolator
from ctapipe.reco.shower_max import ShowerMaxEstimator
import matplotlib.pyplot as plt
from ctapipe.image import poisson_likelihood
from ctapipe.io.containers import ReconstructedShowerContainer, ReconstructedEnergyContainer
from ctapipe.coordinates import HorizonFrame, NominalFrame, TiltedGroundFrame, GroundFrame, project_to_ground
from ctapipe.reco.reco_algorithms import RecoShowerGeomAlgorithm


class ImPACTFitter(RecoShowerGeomAlgorithm):
    """
    This class is an implementation if the ImPACT Monte Carlo Template based image fitting
    method from:
    Parsons & Hinton, Astroparticle Physics 56 (2014), pp. 26-34
    This method uses a comparision of the predicted image from a library of image templates
    to perform a maximum likelihood fit for the shower axis, energy and height of maximum.

    Because this application is computationally intensive the usual advice to use astropy units
    for all quantities is ignored (as these slow down some computations), instead units within the
    class are fixed:
    - Angular units in radians
    - Distance units in metres
    - Energy units in TeV

    """
    def __init__(self, fit_xmax=True, root_dir="/Users/dparsons/Documents/Unix/CTA/ImPACT_pythontests/"):

        # First we create a dictionary of image template interpolators for each telescope type
        self.root_dir = root_dir
        self.prediction = dict()
        self.file_names = {"GATE":"SST-GCT.table.gz", "LSTCam":"LST.table.gz",
                           "NectarCam":"MST.table.gz", "FlashCam":"MST.table.gz"}

        # We also need a conversion function from height above ground to depth of maximum
        # To do this we need the conversion table from CORSIKA
        self.shower_max = ShowerMaxEstimator(root_dir+"atmprof.dat")

        # For likelihood calculation we need the with of the pedestal distribution for each pixel
        # currently this is not availible from the calibration, so for now lets hard code it in a dict
        self.ped_table = {"LSTCam": 1.3, "NectarCam": 1.3, "FlashCam": 2.3,"GATE": 0.5}
        self.spe = 0.5 # Also hard code single p.e. distribution width

        # Also we need to scale the ImPACT templates a bit, this will be fixed later
        self.scale = {"LSTCam": 1., "NectarCam": 1., "FlashCam": 1.0, "GATE": 1.0}

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

        self.fit_xmax = fit_xmax
        self.ped = dict()

        self.array_direction = 0

    def initialise_templates(self, tel_type):
        """
        Check if templates for a given telescope type has been initialised and if not
        do it and add to the dictionary

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
                TableInterpolator(self.root_dir + self.file_names[tel_type[t]])

        return True

    def get_brightest_mean(self, num_pix=3):
        """
        This is a simple function to find the peak position of each image in an event
        which will be used later in the Xmax calculation. Peak is found by taking the
        average position of the n hottest pixels in the image.

        Parameters
        ----------
        num_pix: int
            Number of pixels the average position from

        Returns
        -------
            None
        """
        peak_x = np.zeros([len(self.pixel_x)]) # Create blank arrays for peaks rather than a dict (faster)
        peak_y = np.zeros(peak_x.shape)
        peak_amp = np.zeros(peak_x.shape)

        # Loop over all tels to take weighted average of pixel positions
        # This loop could maybe be replaced by an array operation by a numpy wizard

        tel_num = 0
        for tel in self.image:
            top_index = self.image[tel].argsort()[-1*num_pix:][::-1]
            weight = self.image[tel][top_index]
            weighted_x = self.pixel_x[tel][top_index] * weight
            weighted_y = self.pixel_y[tel][top_index] * weight

            ppx = np.sum(weighted_x)/np.sum(weight)
            ppy = np.sum(weighted_y)/np.sum(weight)

            peak_x[tel_num] = ppx # Fill up array
            peak_y[tel_num] = ppy
            peak_amp[tel_num] = np.sum(weight)
            tel_num+=1

        self.peak_x = peak_x #* unit # Add to class member
        self.peak_y = peak_y #* unit
        self.peak_amp = peak_amp

    # This function would be useful elsewhere so probably be implemented in a more general form
    def get_shower_max(self, source_x, source_y, core_x, core_y, zen):
        """
        Function to calculate the depth of shower maximum geometrically under the assumption
        that the shower maximum lies at the brightest point of the camera image.

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

        # Calculate displacement of image centroid from source position (in rad)
        disp = np.sqrt(np.power(self.peak_x-source_x, 2) +
                       np.power(self.peak_y-source_y, 2))
        # Calculate impact parameter of the shower
        impact = np.sqrt(np.power(np.array(list(self.tel_pos_x.values()))-core_x, 2) +
                         np.power(np.array(list(self.tel_pos_y.values()))-core_y, 2))

        height = impact / disp  # Distance above telescope is ration of these two (small angle)
        weight = np.sqrt(self.peak_amp)  # weight average by amplitude

        # Take weighted mean of esimates
        mean_height = np.sum(height*weight)/np.sum(weight)
        # This value is height above telescope in the tilted system, we should convert to height above ground
        mean_height *= np.cos(zen)
        # Add on the height of the detector above sea level
        mean_height += 2100

        if mean_height > 100000 or np.isnan(mean_height):
            mean_height = 100000

        mean_height *= u.m
        # Lookup this height in the depth tables, the convert Hmax to Xmax
        x_max = self.shower_max.interpolate(mean_height.to(u.km))
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

        pixel_pos_trans_x = (pixel_pos_x-x_trans) * np.cos(phi) - (pixel_pos_y-y_trans) * np.sin(phi)
        pixel_pos_trans_y = (pixel_pos_x-x_trans) * np.sin(phi) + (pixel_pos_y-y_trans) * np.cos(phi)
        return pixel_pos_trans_x, pixel_pos_trans_y

    def image_prediction(self, type, zenith, azimuth, energy, impact, x_max, pix_x, pix_y):
        """
        Creates predicted image for the specified pixels, interpolated from the template library.

        Parameters
        ----------
        type: string
            Telescope type specifier
        zenith: float
            Zenith angle of observations
        azimuth: float
            Azimuth angle of observations
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

        return self.prediction[type].interpolate([energy,impact,x_max], pix_x, pix_y)

    def get_prediction(self, tel_id, shower_reco, energy_reco):


        horizon_seed = HorizonFrame(az=shower_reco.az, alt=shower_reco.alt)
        nominal_seed = horizon_seed.transform_to(NominalFrame(array_direction=self.array_direction))
        source_x = nominal_seed.x.to(u.rad).value
        source_y = nominal_seed.y.to(u.rad).value

        ground = GroundFrame(x=shower_reco.core_x, y=shower_reco.core_y, z=0*u.m)
        tilted = ground.transform_to(TiltedGroundFrame(pointing_direction=self.array_direction))
        tilt_x = tilted.x.to(u.m).value
        tilt_y = tilted.y.to(u.m).value

        zenith = 90*u.deg - self.array_direction[0]
        azimuth = self.array_direction[1]

        #x_max_exp = 343 + 84 * np.log10(energy_reco.energy.value)
        x_max_exp = 300 + 93 * np.log10(energy_reco.energy.value)

        x_max = shower_reco.h_max / np.cos(zenith)
        #x_max = self.shower_max.interpolate(h_max.to(u.km))
        #x_max = x_max_exp
        # Convert to binning of Xmax, addition of 100 can probably be removed
        x_max_bin = x_max.value-x_max_exp
        if x_max_bin > 150:
            x_max_bin = 150
        if x_max_bin < -150:
            x_max_bin = -150

        impact = np.sqrt(pow(self.tel_pos_x[tel_id] - tilt_x, 2) + pow(self.tel_pos_y[tel_id] -
                                                                                   tilt_y, 2))

        phi = np.arctan2((self.tel_pos_y[tel_id] - tilt_y), (self.tel_pos_x[tel_id] - tilt_x))

        pix_x_rot, pix_y_rot = self.rotate_translate(self.pixel_x[tel_id]*-1, self.pixel_y[tel_id],
                                                     source_x, source_y, phi)
        #pix_y_rot = pix_y_rot * -1
        #pix_x_rot = pix_x_rot * -1

        prediction = self.image_prediction(self.type[tel_id], 20*u.deg, 0*u.deg,
                                           energy_reco.energy.value, impact, x_max_bin,
                                           pix_x_rot*(180/math.pi), pix_y_rot*(180/math.pi))

        prediction *= self.scale[self.type[tel_id]]
        prediction[prediction < 0] = 0
        prediction[np.isnan(prediction)] = 0

        return prediction

    def get_likelihood(self, source_x, source_y, core_x, core_y, energy, x_max_scale):
        """
        Get the likelihood that the image predicted at the given test position matches the
        camera image.

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

        # First we add units back onto everything.
        # Currently not handled very well, maybe in future we could just put everything in the correct units
        # when loading in the class and ignore them from then on

        zenith = 90*u.deg - self.array_direction[0]
        azimuth = self.array_direction[1]
        # Geometrically calculate the depth of maximum given this test position

        if self.fit_xmax:
            x_max = self.get_shower_max(source_x, source_y,
                                        core_x, core_y,
                                        zenith.to(u.rad).value) * x_max_scale

            # Calculate expected Xmax given this energy
            #x_max_exp = 343 + 84 * np.log10(energy)
            x_max_exp = 300 + 93 * np.log10(energy)

            # Convert to binning of Xmax, addition of 100 can probably be removed
            x_max_bin = x_max.value - x_max_exp
            #x_max_bin = 100

            # Check for range
            if x_max_bin > 150:
                x_max_bin = 150
            if x_max_bin < -150:
                x_max_bin = -150
        else:
            x_max_bin = x_max_scale * 37.8
            if x_max_bin< 13:
                x_max_bin = 13

        sum_like = 0
        for tel_count in self.image:  # Loop over all telescopes
            # Calculate impact distance for all telescopes
            impact = np.sqrt(pow(self.tel_pos_x[tel_count] - core_x, 2) + pow(self.tel_pos_y[tel_count] - core_y, 2))
            # And the expected rotation angle
            phi = np.arctan2((self.tel_pos_y[tel_count] - core_y), (self.tel_pos_x[tel_count] - core_x))# - (math.pi/2.)

            # Rotate and translate all pixels such that they match the template orientation
            pix_x_rot, pix_y_rot = self.rotate_translate(self.pixel_x[tel_count]*-1, self.pixel_y[tel_count],
                                                         source_x, source_y, phi)

            #plt.plot(pix_x_rot, pix_y_rot)
            #plt.show()
            # Then get the predicted image, convert pixel positions to deg
            prediction = self.image_prediction(self.type[tel_count], zenith, azimuth,
                                               energy, impact, x_max_bin,
                                               pix_x_rot*(180/math.pi), pix_y_rot*(180/math.pi))
            prediction[np.isnan(prediction)] = 0
            prediction[prediction<0] = 0

            # Scale templates to match simulations
            prediction *= self.scale[self.type[tel_count]]
            # Get likelihood that the prediction matched the camera image
            like = poisson_likelihood(self.image[tel_count], prediction, self.spe, self.ped[tel_count])
            like[np.isnan(like)] = 1e9
            sum_like += np.sum(like)
            if np.sum(prediction) is 0:
                sum_like += 1e9

        return sum_like

    def get_likelihood_min(self, x):
        """
        Wrapper class around likelihood function for use with scipy minimisers

        Parameters
        ----------
        x: ndarray
            Array of minimisation parameters

        Returns
        -------
        float: Likelihood value of test position
        """
        return self.get_likelihood(x[0], x[1], x[2], x[3], x[4], x[5])

    def set_event_properties(self, image, pixel_x, pixel_y, pixel_area, type_tel, tel_x, tel_y, array_direction):
        """
        The setter class is used to set the event properties within this class before minimisation
        can take place. This simply copies a bunch of useful properties to class members, so that we
        can use them later without passing all this information around.

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
        # First store these parameters in the class so we can use them in minimisation
        # For most values this is simply copying
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
            self.pixel_area[x] = pixel_area[x].value
            self.ped[x] = self.ped_table[type_tel[x]] # Here look up pedestal value

        self.get_brightest_mean(num_pix=3)
        self.type = type_tel
        self.initialise_templates(type_tel)

        self.array_direction= array_direction

    def predict(self, shower_seed, energy_seed):
        """

        Parameters
        ----------
        source_x: float
            Initial guess of source position in the nominal frame
        source_y: float
            Initial guess of source position in the nominal frame
        core_x: float
            Initial guess of the core position in the tilted system
        core_y: float
            Initial guess of the core position in the tilted system
        energy: float
            Initial guess of energy

        Returns
        -------
        Shower object with fit results
        """
        horizon_seed = HorizonFrame(az=shower_seed.az, alt=shower_seed.alt)
        nominal_seed = horizon_seed.transform_to(NominalFrame(array_direction=self.array_direction))

        source_x = nominal_seed.x.to(u.rad).value
        source_y = nominal_seed.y.to(u.rad).value

        ground = GroundFrame(x=shower_seed.core_x, y=shower_seed.core_y, z=0*u.m)
        tilted = ground.transform_to(TiltedGroundFrame(pointing_direction=self.array_direction))
        tilt_x = tilted.x.to(u.m).value
        tilt_y = tilted.y.to(u.m).value

        lower_en_limit = energy_seed.energy*0.1
        if lower_en_limit < 0.04*u.TeV:
            lower_en_limit = 0.04*u.TeV
        # Create Minuit object with first guesses at parameters, strip away the units as Minuit doesnt like them
        min = Minuit(self.get_likelihood, print_level=1,
                     source_x=source_x, error_source_x=0.01/57.3, fix_source_x=False,
                     limit_source_x= (source_x - 0.5/57.3, source_x + 0.5 / 57.3),
                     source_y=source_y, error_source_y=0.01/57.3, fix_source_y=False,
                     limit_source_y=(source_y - 0.5 / 57.3, source_y+ 0.5 / 57.3),
                     core_x=tilt_x, error_core_x=10, limit_core_x=(tilt_x-200, tilt_x+200),
                     core_y=tilt_y, error_core_y=10, limit_core_y=(tilt_y-200,tilt_y+200),
                     energy=energy_seed.energy.value, error_energy=energy_seed.energy.value*0.05,
                     limit_energy=(lower_en_limit.value,energy_seed.energy.value*10.),
                     x_max_scale=1, error_x_max_scale=0.1, limit_x_max_scale=(0.5,2), fix_x_max_scale=False, errordef=1)

        min.tol *= 1000
        min.strategy = 0

        # Perform minimisation
        migrad = min.migrad()
        fit_params = min.values
        errors = min.errors
    #    print(migrad)
    #    print(min.minos())

        # container class for reconstructed showers '''
        shower_result = ReconstructedShowerContainer()

        nominal = NominalFrame(x=fit_params["source_x"] * u.rad, y=fit_params["source_y"] * u.rad,
                               array_direction=self.array_direction)
        horizon = nominal.transform_to(HorizonFrame())

        shower_result.alt, shower_result.az = horizon.alt, horizon.az
        tilted = TiltedGroundFrame(x=fit_params["core_x"] * u.m, y=fit_params["core_y"] * u.m,
                                   pointing_direction=self.array_direction)
        ground = project_to_ground(tilted)

        shower_result.core_x = ground.x
        shower_result.core_y = ground.y

        shower_result.is_valid = True

        shower_result.alt_uncert = np.nan
        shower_result.az_uncert = np.nan
        shower_result.core_uncert = np.nan
        zenith = 90*u.deg - self.array_direction[0]
        shower_result.h_max = fit_params["x_max_scale"] * \
                              self.get_shower_max(fit_params["source_x"],fit_params["source_y"],
                                                  fit_params["core_x"],fit_params["core_y"],
                                                  zenith.to(u.rad).value)
        shower_result.h_max_uncert = errors["x_max_scale"] * shower_result.h_max
        shower_result.goodness_of_fit = np.nan
        shower_result.tel_ids = list(self.image.keys())

        energy_result = ReconstructedEnergyContainer()
        energy_result.energy = fit_params["energy"] * u.TeV
        energy_result.energy_uncert = errors["energy"] * u.TeV
        energy_result.is_valid = True
        energy_result.tel_ids = list(self.image.keys())
        # Return interesting stuff

        return shower_result, energy_result

    # These drawing functions should really be moved elsewhere, as drawing a 2D map of the array is quite
    # generic and should probably live in plotting
    def draw_surfaces(self, x_src, y_src, x_grd, y_grd, energy, xmax):
        """
        Simple function to draw the surface of the test statistic in both the nominal
        and tilted planes while keeping the values in the other plane fixed at the best fit
        value.

        Parameters
        ----------
        x_src: float
            Source position in nominal coordinates (centre of map)
        y_src: float
            Source position in nominal coordinates (centre of map)
        x_grd: float
            Ground position in tilted coordinates (centre of map)
        y_grd: float
            Ground position in tilted coordinates (centre of map)

        Returns
        -------

        """
        fig = plt.figure(figsize=(12, 6))
        nom1 = fig.add_subplot(121)
        self.draw_nominal_surface(x_src, y_src, x_grd, y_grd, energy, xmax, nom1, bins=30, range=0.5*u.deg)
        nom1.plot(x_src, y_src, "wo")

        tilt1 = fig.add_subplot(122)
        self.draw_tilted_surface(x_src, y_src, x_grd, y_grd, energy, xmax, tilt1, bins=30, range=100*u.m)
        tilt1.plot(x_grd, y_grd, "wo")

        plt.show()

        return

    def draw_nominal_surface(self, x_src, y_src, x_grd, y_grd, energy, xmax, plot_name, bins=30, range=1):
        """
        Function for creating test statistic surface in nominal plane

        Parameters
        ----------
        x_src: float
            Source position in nominal coordinates (centre of map)
        y_src: float
            Source position in nominal coordinates (centre of map)
        x_grd: float
            Ground position in tilted coordinates (centre of map)
        y_grd: float
            Ground position in tilted coordinates (centre of map)
        plot_name: matplotlib axis
            Subplot in which to include this plot
        bins: int
            Number of bins in each axis
        range: float
            Size of map

        Returns
        -------
            None
        """
        x_dir = np.linspace(x_src - range, x_src + range, num=bins)
        y_dir = np.linspace(y_src - range, y_src + range, num=bins)
        w = np.zeros([bins,bins])

        i=0
        for xb in x_dir:
            j = 0
            for yb in y_dir:
                w[i][j] = self.get_likelihood(xb.to(u.rad).value, yb.to(u.rad).value, x_grd.value, y_grd.value, energy.value,
                                              xmax)
                j += 1
            i += 1

        return plot_name.imshow(w, interpolation="nearest", cmap=plt.cm.viridis_r,
                                extent=(x_src.value - range.value, x_src.value + range.value,
                                         y_src.value - range.value, y_src.value + range.value))

    def draw_tilted_surface(self, x_src, y_src, x_grd, y_grd, energy, xmax, bins=50, range=100*u.m):
        """
        Function for creating test statistic surface in tilted plane

        Parameters
        ----------
        x_src: float
            Source position in nominal coordinates (centre of map)
        y_src: float
            Source position in nominal coordinates (centre of map)
        x_grd: float
            Ground position in tilted coordinates (centre of map)
        y_grd: float
            Ground position in tilted coordinates (centre of map)
        plot_name: matplotlib axis
            Subplot in which to include this plot
        bins: int
            Number of bins in each axis
        range: float
            Size of map

        Returns
        -------
            None
        """
        x_ground_list = np.linspace(x_grd - range, x_grd + range, num=bins)
        y_ground_list = np.linspace(y_grd - range, y_grd + range, num=bins)
        w = np.zeros([bins,bins])

        i = 0
        for xb in x_ground_list:
            j = 0
            for yb in y_ground_list:
                w[i][j] = self.get_likelihood(x_src.to(u.rad).value, y_src.to(u.rad).value,
                                              xb.value, yb.value, energy.value, xmax)
                j += 1

            i += 1

        X, Y = np.meshgrid(x_ground_list, y_ground_list)
        return X, Y, w
        #plot_name.imshow(w, interpolation="nearest", cmap=plt.cm.viridis_r,
                                #extent=(x_grd.value - range.value , x_grd.value + range.value,
                                #        y_grd.value -range.value, y_grd.value + range.value))

