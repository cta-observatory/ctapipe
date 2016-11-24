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
from scipy.optimize import minimize


class ImPACTFitter(object):
    """
    This class is an implementation if the ImPACT Monte Carlo Template based image fitting
    method from:
    Parsons & Hinton, Astroparticle Physics 56 (2014), pp. 26-34
    This method uses a comparision of the predicted image from a library of image templates
    to perform a maximum likelihood fit for the shower axis, energy and height of maximum.
    """
    def __init__(self):

        # First we create a dictionary of image template interpolators for each telescope type
        self.prediction = dict()
        self.prediction["LSTCam"] = \
            TableInterpolator("/Users/dparsons/Documents/Unix/CTA/ImPACT_pythontests/LST.obj")
        self.prediction["NectarCam"] = \
            TableInterpolator("/Users/dparsons/Documents/Unix/CTA/ImPACT_pythontests/MST_NectarCam.obj")
        self.prediction["GATE"] = \
            TableInterpolator("/Users/dparsons/Documents/Unix/CTA/ImPACT_pythontests/SST_GCT.obj")

        # We also need a conversion function from height above ground to depth of maximum
        # To do this we need the conversion table from CORSIKA
        self.shower_max = ShowerMaxEstimator("/Users/dparsons/Documents/Unix/CTA/ImPACT_pythontests/atmprof.dat")

        # For likelihood calculation we need the with of the pedestal distribution for each pixel
        # currently this is not availible from the calibration, so for now lets hard code it in a dict
        self.ped_table = {"LSTCam": 1.3, "NectarCam": 1.3, "GATE": 0.8}
        self.spe = 0.5 # Also hard code single p.e. distribution width
        self.ped = 0

        # Also we need to scale the ImPACT templates a bit, this will be fixed later
        self.scale = {"LSTCam": 1.2, "NectarCam": 1.1, "GATE": 0.75}

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

        self.unit = u.deg


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
        peak_x = np.zeros([len(self.pixel_x)]) # Create blank arrays for peaks
        peak_y = np.zeros(peak_x.shape)
        peak_amp = np.zeros(peak_x.shape)
        unit = self.pixel_x[0].unit

        # Loop over all tels to take weighted average of pixel positions
        # This loop could maybe be replaced by an array operation by a numpy wizard
        for im, px, py, tel_num in zip(self.image, self.pixel_x, self.pixel_y, range(len(self.pixel_y))):
            top_index = im.argsort()[-1*num_pix:][::-1]
            weight = im[top_index]
            weighted_x = px[top_index] * weight
            weighted_y = py[top_index] * weight

            ppx = np.sum(weighted_x)/np.sum(weight)
            ppy = np.sum(weighted_y)/np.sum(weight)

            peak_x[tel_num] = ppx.value # Fill up array
            peak_y[tel_num] = ppy.value
            peak_amp[tel_num] = np.sum(weight)

        self.peak_x = peak_x * unit # Add to class member
        self.peak_y = peak_y * unit
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
        disp = np.sqrt(np.power(self.peak_x.to(u.rad).value-source_x, 2) +
                       np.power(self.peak_y.to(u.rad).value-source_y, 2))
        # Calculate impact parameter of the shower
        impact = np.sqrt(np.power(self.tel_pos_x.to(u.m).value-core_x, 2) +
                         np.power(self.tel_pos_y.to(u.m).value-core_y, 2))

        height = impact / disp  # Distance above telescope is ration of these two (small angle)
        weight = np.sqrt(self.peak_amp)  # weight average by amplitude

        # Take weighted mean of esimates
        mean_height = np.sum(height*weight)/np.sum(weight)
        # This value is height above telescope in the tilted system, we should convert to height above ground
        mean_height *= np.cos(zen)
        # Add on the height of the detector above sea level
        mean_height += 2100

        if mean_height > 100000:
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

        return self.prediction[type].interpolate([energy.to(u.TeV).value,impact.to(u.m).value,x_max],
                                                 pix_x, pix_y)

    def get_prediction(self, tel, zenith, azimuth, core_x, core_y, energy, x_max_scale):
        """

        Parameters
        ----------
        tel
        zenith
        azimuth
        core_x
        core_y
        energy
        x_max_scale

        Returns
        -------

        """
        source_x = 0*u.deg
        source_y = 0*u.deg
        x_max = self.get_shower_max(source_x.to(u.rad).value, source_y.to(u.rad).value,
                                    core_x.to(u.m).value, core_y.to(u.m).value,
                                    zenith.to(u.rad).value) * x_max_scale

        x_max_exp = 300 + 93 * np.log10(energy.value)

        x_max_bin = 100 + (x_max.value - x_max_exp) / 25
        if x_max_bin > 107:
            x_max_bin = 107
        if x_max_bin < 93:
            x_max_bin = 93
        impact = np.sqrt(pow(self.tel_pos_x[tel] - core_x, 2) + pow(self.tel_pos_y[tel] - core_y, 2))
        phi = np.arctan2((self.tel_pos_x[tel] - core_x), (self.tel_pos_y[tel] - core_y))
        phi += 180 * u.deg

        print(energy, impact,x_max_bin,self.type[tel],self.pixel_area[tel].to(u.deg * u.deg).value)

        pix_x_rot, pix_y_rot = self.rotate_translate(self.pixel_x[tel], self.pixel_y[tel],
                                                     source_x, source_y, phi.to(u.rad))

        pix_y_rot += 0.02 * u.deg

        prediction = self.image_prediction(self.type[tel], zenith, azimuth,
                                           energy, impact, x_max_bin,
                                           pix_x_rot, pix_y_rot)
        prediction *= self.pixel_area[tel].to(u.deg * u.deg).value
        prediction *= self.scale[self.type[tel]]
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
        x_max_scale:
            Scaling factor applied to geometrically calculated Xmax

        Returns
        -------
        float: Likelihood the model represents the camera image at this position

        """

        # First we add units back onto everything.
        # Currently not handled very well, maybe in future we could just put everything in the correct units
        # when loading in the class and ignore them from then on
        source_x *= u.deg
        source_y *= u.deg

        core_x *= u.m
        core_y *= u.m

        energy *= u.TeV

        # Hardcoded for now, will be a coordinate transform later
        zenith = 20*u.deg
        azimuth = 0*u.deg
        # Geometrically calculate the depth of maximum given this test position
        x_max = self.get_shower_max(source_x.to(u.rad).value, source_y.to(u.rad).value,
                                    core_x.to(u.m).value, core_y.to(u.m).value,
                                    zenith.to(u.rad).value) * x_max_scale
        # Calculate expected Xmax given this energy
        x_max_exp = 300 + 93 * np.log10(energy.value)
        # Convert to binning of Xmax, addition of 100 can probably be removed
        x_max_bin = 100 + (x_max.value - x_max_exp)/25
        # Check for range
        if x_max_bin > 107:
            x_max_bin = 107
        if x_max_bin < 93:
            x_max_bin = 93

        # Calculate impact distance for all telescopes
        impact = np.sqrt(pow(self.tel_pos_x - core_x, 2) + pow(self.tel_pos_y - core_y, 2))
        # And the expected rotation angle
        phi = np.arctan2((self.tel_pos_x - core_x), (self.tel_pos_y - core_y))
        phi += 180 * u.deg # Can't explain why this is needed!

        sum_like = 0
        for tel_count in range(self.image.shape[0]):  # Loop over all telescopes
            # Rotate and translate all pixels such that they match the template orientation
            pix_x_rot, pix_y_rot = self.rotate_translate(self.pixel_x[tel_count], self.pixel_y[tel_count],
                                                         source_x, source_y, phi[tel_count])

            pix_y_rot += 0.02 * u.deg  # This needs to be added to correct for biasing in the smoothing
            pix_x_rot += 0.02 * u.deg

            # Then get the predicted image
            prediction = self.image_prediction(self.type[tel_count], zenith, azimuth,
                                               energy, impact[tel_count], x_max_bin,
                                               pix_x_rot, pix_y_rot)
            # Scale templates to match simulations
            prediction *= self.scale[self.type[tel_count]]
            prediction *= self.pixel_area[tel_count].to(u.deg*u.deg).value  # Scale by pixel area
            # Get likelihood that the prediction matched the camera image
            like = self.calc_likelihood(self.image[tel_count], prediction, self.spe, self.ped[tel_count])
            sum_like += np.sum(like)

        return -2 * sum_like  # Multiply by -2 to make is chi-squared like

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


    @staticmethod
    def calc_likelihood(image, prediction, spe_width, ped):
        """
        Calculate likelihood of prediction given the measured signal, gaussian approx from
        de Naurois et al 2009

        Parameters
        ----------
        image: ndarray
            Pixel amplitudes from image
        prediction: ndarray
            Predicted pixel amplitudes from model
        spe_width: ndarray
            width of single p.e. distributio
        ped: ndarray
            width of pedestal

        Returns
        -------
        ndarray: likelihood for each pixel
        """
        sq = 1./np.sqrt(2 * math.pi * (np.power(ped, 2)
                                       + prediction * (1 + np.power(spe_width, 2))))

        diff = np.power(image - prediction, 2.)
        denom = 2 * (np.power(ped, 2) + prediction * (1 + np.power(spe_width, 2)))
        expo = np.exp(-1 * diff / denom)
        sm = expo<1e-300
        expo[sm] = 1e-300

        return np.log(sq*expo)

    def set_event_properties(self, image, pixel_x, pixel_y, pixel_area, type_tel, tel_x, tel_y):
        """
        The setter class is used to set the event properties within this class before minimisation
        can take place. This simply copies a bunch of useful properties to class members, so that we
        can use them later without passing all this information around.

        Parameters
        ----------
        image: ndarray
            Amplitude of pixels in camera images
        pixel_x: list
            X position of pixels in nominal system
        pixel_y: list
            Y position of pixels in nominal system
        pixel_area: list
            Area of pixel in each telescope type
        type_tel: list
            Type of telescope
        tel_x: list
            X position of telescope
        tel_y: list
            Y position of telescope

        Returns
        -------
        None
        """
        # First store these parameters in the class so we can use them in minimisation
        # For most values this is simply copying
        self.image = np.asarray(image)

        self.pixel_x = pixel_x
        self.pixel_y = pixel_y

        self.unit = pixel_x[0].unit

        # However in this case we need the value to be in a numpy array,
        # so we have t be careful how we copy it
        tel_unit = tel_x[0].unit
        tel_x_array = np.zeros(len(tel_x))
        tel_y_array = np.zeros(len(tel_x))

        pixel_area_array = np.zeros(len(tel_x))
        area_unit = pixel_area[0].unit
        ped = np.zeros(len(tel_x))
        # So here we must loop over the telescopes
        for x in range(len(tel_x)):
            tel_x_array[x] = tel_x[x].value
            tel_y_array[x] = tel_y[x].value
            pixel_area_array[x] = pixel_area[x].value
            ped[x] = self.ped_table[type_tel[x]] # Here look up pedestal value

        self.pixel_area = pixel_area_array * area_unit

        self.tel_pos_x = tel_x_array * tel_unit
        self.tel_pos_y = tel_y_array * tel_unit

        self.get_brightest_mean(num_pix=5)
        self.type = type_tel
        self.ped = ped

    def fit_event(self, source_x, source_y, core_x, core_y, energy):
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

        # Create Minuit object with first guesses at parameters, strip away the units as Minuit doesnt like them
        min = Minuit(self.get_likelihood, print_level=1,
                     source_x=source_x.value, error_source_x=0.01, fix_source_x=False,
                     source_y=source_y.value, error_source_y=0.01, fix_source_y=False,
                     core_x=core_x.value, error_core_x=10, limit_core_x= (core_x.value-200,core_x.value+200), fix_core_x=False,
                     core_y=core_y.value, error_core_y=10, limit_core_y= (core_y.value-200,core_y.value+200), fix_core_y=False,
                     energy=energy.value, error_energy=energy.value*0.05, limit_energy=(energy.value*0.1,energy.value*10.),
                     x_max_scale=1, error_x_max_scale=0.05, limit_x_max_scale=(0.5,1.5), errordef=1)

        min.tol *= 1000
        min.strategy = 0

        # Perform minimisation
        migrad = min.migrad()
        fit_params = min.values
        print(fit_params)
        print(migrad)

        #self.draw_surfaces(fit_params["source_x"]*u.deg, fit_params["source_y"]*u.deg,
        #                   fit_params["core_x"]*u.m, fit_params["core_y"]*u.m,
        #                   fit_params["energy"]*u.TeV, fit_params["x_max_scale"])

        # Return interesting stuff
        return fit_params

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

    def draw_nominal_surface(self, x_src, y_src, x_grd, y_grd, energy, xmax, plot_name, bins=100, range=1):
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
                w[i][j] = self.get_likelihood(xb.value, yb.value, x_grd.value, y_grd.value, energy.value, xmax)
                j += 1
            i += 1

        return plot_name.imshow(w, interpolation="nearest", cmap=plt.cm.viridis_r,
                                extent=(x_src.value - range.value, x_src.value + range.value,
                                         y_src.value - range.value, y_src.value + range.value))

    def draw_tilted_surface(self, x_src, y_src, x_grd, y_grd, energy, xmax, plot_name, bins=100, range=100):
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
                w[i][j] = self.get_likelihood(x_src.value, y_src.value, xb.value, yb.value, energy.value, xmax)
                j += 1

            i += 1

        return plot_name.imshow(w, interpolation="nearest", cmap=plt.cm.viridis_r,
                                extent=(x_grd.value - range.value , x_grd.value + range.value,
                                        y_grd.value -range.value, y_grd.value + range.value))

