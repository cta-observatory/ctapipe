import random

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import ICRS, AltAz, SkyCoord
from astropy.time import Time
from scipy.odr import ODR, Model, RealData

from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.coordinates.utils import cart2pol


class StarTracker:
    """
    Utility class to provide the position of the star in the telescope's camera frame coordinates at a given time
    """

    def __init__(
        self,
        star_label,
        star_coordinates,
        telescope_location,
        telescope_focal_length,
        telescope_pointing,
        observed_wavelength,
        relative_humidity,
        temperature,
        pressure,
        pointing_label=None,
    ):
        """
        Constructor

        :param str star_label: Star label
        :param SkyCoord star_coordinates: Star coordinates in ICRS frame
        :param EarthLocation telescope_location: Telescope location coordinates
        :param Quantity[u.m] telescope_focal_length: Telescope focal length [m]
        :param SkyCoord telescope_pointing: Telescope pointing in ICRS frame
        :param Quantity[u.micron] observed_wavelength: Telescope focal length [micron]
        :param float relative_humidity: Relative humidity
        :param Quantity[u.deg_C] temperature: Temperature [C]
        :param Quantity[u.hPa] pressure: Pressure [hPa]
        :param str pointing_label: Pointing label
        """
        self.star_label = star_label
        self.star_coordinates_icrs = star_coordinates
        self.telescope_location = telescope_location
        self.telescope_focal_length = telescope_focal_length
        self.telescope_pointing = telescope_pointing
        self.obswl = observed_wavelength
        self.relative_humidity = relative_humidity
        self.temperature = temperature
        self.pressure = pressure
        self.pointing_label = pointing_label

    def position_in_camera_frame(self, timestamp, pointing=None, focal_correction=0):
        """
        Calculates star position in the engineering camera frame

        :param astropy.Time timestamp: Timestamp of the observation
        :param SkyCoord pointing: Current telescope pointing in ICRS frame
        :param float focal_correction: Correction to the focal length of the telescope. Float, should be provided in meters

        :return: Pair (float, float) of star's (x,y) coordinates in the engineering camera frame in meters
        """
        # If no telescope pointing is provided, use the telescope pointing, provided
        # during the class member initialization
        if pointing is None:
            pointing = self.telescope_pointing
        # Determine current telescope pointing in AltAz
        altaz_pointing = pointing.transform_to(
            AltAz(
                obstime=timestamp,
                location=self.telescope_location,
                obswl=self.obswl,
                relative_humidity=self.relative_humidity,
                temperature=self.temperature,
                pressure=self.pressure,
            )
        )
        # Create current camera frame
        camera_frame = EngineeringCameraFrame(
            telescope_pointing=altaz_pointing,
            focal_length=self.telescope_focal_length + focal_correction * u.m,
            obstime=timestamp,
            location=self.telescope_location,
        )
        # Calculate the star's coordinates in the current camera frame
        star_coords_camera = self.star_coordinates_icrs.transform_to(camera_frame)
        return (star_coords_camera.x.to_value(), star_coords_camera.y.to_value())


class StarFitter:
    """
    Star trajectory fitting class
    """

    def __init__(
        self,
        stars,
        telescope_pointing,
        telescope_location,
        focal_length,
        observed_wavelength,
        relative_humidity,
        temperature,
        pressure,
        fit_grid="polar",
    ):
        """
        Constructor

        :param stars: Astropy table containing the star coordinates
        :param SkyCoord telescope_pointing: Telescope pointing in ICRS frame
        :param EarthLocation telescope_location: Telescope location
        :param Quantity[u.m] telescope_focal_length: Telescope focal length [m]
        :param Quantity[u.micron] observed_wavelength: Telescope focal length [micron]
        :param float relative_humidity: Relative humidity
        :param Quantity[u.deg_C] temperature: Temperature [C]
        :param Quantity[u.hPa] pressure: Pressure [hPa]
        :param str fit_grid: Coordinate system grid to use. Either polar or cartesian
        """
        self.stars = stars
        self.telescope_pointing = telescope_pointing
        self.telescope_location = telescope_location
        self.focal_length = focal_length
        self.obswl = observed_wavelength
        self.relative_humidity = relative_humidity
        self.temperature = temperature
        self.pressure = pressure
        self.stars = []
        for star in stars:
            self.stars.append(self.init_star(star))
        self.fit_mode = "xy"
        self.fit_grid = fit_grid
        self.star_motion_model = Model(self.fit_function)
        self.fit_summary = None
        self.fit_resuts = None

    def init_star(self, star):
        """
        Initialize StarTracker object for a given star

        :param str star_label: Star label according to NOMAD catalog

        :return: StarTracker object
        """

        star_coords = SkyCoord(
            star["ra_dec"].ra,
            star["ra_dec"].dec,
            unit="deg",
            frame="icrs",
            obswl=self.obswl,
            relative_humidity=self.relative_humidity,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        st = StarTracker(
            star["Name"],
            star_coords,
            self.telescope_location,
            self.focal_length,
            self.telescope_pointing,
            self.obswl,
            self.relative_humidity,
            self.temperature,
            self.pressure,
        )
        return st

    def current_pointing(self, t):
        """
        Retrieve current telescope pointing
        """
        return self.telescope_pointing

    def fit_function(self, p, t):
        """
        Construct the fit function

        :param p: Fit parameters
        :param array-like(float) t: Timestamp in UNIX_TAI format

        :return: 2D array of star coordinates: [[x_1], [y_1]...[x_N], [y_N]] where array.shape == (N(stars) * 2, len(t))
        """
        time = Time(t, format="unix_tai", scale="utc")
        coord_list = []
        if self.fit_mode == "radec":
            m_ra, m_dec = p
            new_ra = self.current_pointing(time).ra + m_ra * u.deg
            new_dec = self.current_pointing(time).dec + m_dec * u.deg
            new_pointing = SkyCoord(ICRS(ra=new_ra, dec=new_dec))
            for star in self.stars:
                x, y = star.position_in_camera_frame(time, new_pointing)
                if self.fit_grid == "polar":
                    x, y = cart2pol(x, y)
                coord_list.extend([x])
                coord_list.extend([y])
        elif self.fit_mode == "y":
            dy = p
            for star in self.stars:
                x, y = star.position_in_camera_frame(time, self.current_pointing(time))
                y = y + dy
                if self.fit_grid == "polar":
                    x, y = cart2pol(x, y)
                coord_list.extend([x])
                coord_list.extend([y])
        elif self.fit_mode == "xy":
            for star in self.stars:
                dx, dy = p
                x, y = star.position_in_camera_frame(time, self.current_pointing(time))
                x = x + dx
                y = y + dy
                if self.fit_grid == "polar":
                    x, y = cart2pol(x, y)
                coord_list.extend([x])
                coord_list.extend([y])
        elif self.fit_mode == "xyz":
            dx, dy, dz = p
            for star in self.stars:
                x, y = star.position_in_camera_frame(
                    time, self.current_pointing(time), dz
                )
                x = x + dx
                y = y + dy
                if self.fit_grid == "polar":
                    x, y = cart2pol(x, y)
                coord_list.extend([x])
                coord_list.extend([y])
        return np.array(coord_list)

    def generate_mispointed_data(self, mispointing, time_range, random_shift=0):
        """
        Generates mispointed and randomly shifted star positions. Serves for testing and/or illustration purposes.

        :param mispointing: Mispointing [RA, DEC] in degrees
        :param time_range: time range in UNIX_TAI seconds, np.array of astropy.time.Time objects
        :param random_shift: Random position shift in meters
        """
        data = self.fit_function(mispointing, time_range)
        data = np.vectorize(lambda x: x + random.uniform(-random_shift, random_shift))(
            data
        )
        return data

    def fit(self, data, errors, time_range, pointings, fit_mode="xy"):
        """
        Performs the ODR fit of stars trajectories and saves the results as self.fit_results

        :param array data: Reconstructed star positions, data.shape = (N(stars) * 2, len(time_range)), order: x_1, y_1...x_N, y_N
        :param array errors: Uncertainties on the reconstructed star positions. Same shape and order as for the data
        :param array time_range: Array of timestamps in UNIX_TAI format
        :param array-like(SkyCoord) pointings: Array of telescope pointings in ICRS frame
        :param str fit_mode: Fit mode. Can be 'y', 'xy' (default), 'xyz' or 'radec'.
        """
        self.fit_mode = fit_mode
        if self.fit_mode == "radec" or self.fit_mode == "xy":
            init_mispointing = [0, 0]
        elif self.fit_mode == "y":
            init_mispointing = [0]
        elif self.fit_mode == "xyz":
            init_mispointing = [0, 0, 0]
        self.telescope_pointing = pointings
        if errors is not None:
            rdata = RealData(x=time_range, y=data, sy=errors)
        else:
            rdata = RealData(x=time_range, y=data)
        odr = ODR(rdata, self.star_motion_model, beta0=init_mispointing)
        self.fit_summary = odr.run()
        if self.fit_mode == "radec":
            self.fit_results = pd.DataFrame(
                data={
                    "dRA": [self.fit_summary.beta[0]],
                    "dDEC": [self.fit_summary.beta[1]],
                    "eRA": [self.fit_summary.sd_beta[0]],
                    "eDEC": [self.fit_summary.sd_beta[1]],
                }
            )
        elif self.fit_mode == "xy":
            self.fit_results = pd.DataFrame(
                data={
                    "dX": [self.fit_summary.beta[0]],
                    "dY": [self.fit_summary.beta[1]],
                    "eX": [self.fit_summary.sd_beta[0]],
                    "eY": [self.fit_summary.sd_beta[1]],
                }
            )
        elif self.fit_mode == "y":
            self.fit_results = pd.DataFrame(
                data={
                    "dY": [self.fit_summary.beta[0]],
                    "eY": [self.fit_summary.sd_beta[0]],
                }
            )
        elif self.fit_mode == "xyz":
            self.fit_results = pd.DataFrame(
                data={
                    "dX": [self.fit_summary.beta[0]],
                    "dY": [self.fit_summary.beta[1]],
                    "dZ": [self.fit_summary.beta[2]],
                    "eX": [self.fit_summary.sd_beta[0]],
                    "eY": [self.fit_summary.sd_beta[1]],
                    "eZ": [self.fit_summary.sd_beta[2]],
                }
            )
