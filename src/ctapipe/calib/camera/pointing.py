"""
Definition of the `CameraCalibrator` class, providing all steps needed to apply
calibration and image extraction, as well as supporting algorithms.
"""

import copy
from functools import cache

import astropy.units as u
import numpy as np
import pandas as pd
import Vizier  # discuss this dependency with max etc.
from astropy.coordinates import ICRS, AltAz, Angle, EarthLocation, SkyCoord
from astropy.table import QTable
from astropy.time import Time
from scipy.odr import ODR, Model, RealData

from ctapipe.calib.camera.extractor import StatisticsExtractor
from ctapipe.containers import StarContainer
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import (
    ComponentName,
    Dict,
    Float,
    Integer,
    TelescopeParameter,
)
from ctapipe.image import tailcuts_clean
from ctapipe.image.psf_model import PSFModel
from ctapipe.io import EventSource, FlatFieldInterpolator, TableLoader

__all__ = [
    "PointingCalculator",
]


@cache
def _get_pixel_index(n_pixels):
    """Cached version of ``np.arange(n_pixels)``"""
    return np.arange(n_pixels)


def _get_invalid_pixels(n_channels, n_pixels, pixel_status, selected_gain_channel):
    broken_pixels = np.zeros((n_channels, n_pixels), dtype=bool)

    index = _get_pixel_index(n_pixels)
    masks = (
        pixel_status.hardware_failing_pixels,
        pixel_status.pedestal_failing_pixels,
        pixel_status.flatfield_failing_pixels,
    )
    for mask in masks:
        if mask is not None:
            if selected_gain_channel is not None:
                broken_pixels |= mask[selected_gain_channel, index]
            else:
                broken_pixels |= mask

    return broken_pixels


def cart2pol(x, y):
    """
    Convert cartesian coordinates to polar

    :param float x: X coordinate [m]
    :param float y: Y coordinate [m]

    :return: Tuple (r, φ)[m, rad]
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    """
    Convert polar coordinates to cartesian

    :param float rho: R coordinate
    :param float phi: ¢ coordinate [rad]

    :return: Tuple (x,y)[m, m]
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


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


class PointingCalculator(TelescopeComponent):
    """
    Component to calculate pointing corrections from interleaved skyfield events.

    Attributes
    ----------
    stats_extractor: str
        The name of the StatisticsExtractor subclass to be used to calculate the statistics of an image set
    telescope_location: dict
        The location of the telescope for which the pointing correction is to be calculated
    """

    telescope_location = Dict(
        {"longitude": 342.108612, "latitude": 28.761389, "elevation": 2147},
        help="Telescope location, longitude and latitude should be expressed in deg, "
        "elevation - in meters",
    ).tag(config=True)

    observed_wavelength = Float(
        0.35,
        help="Observed star light wavelength in microns"
        "(convolution of blackbody spectrum with camera sensitivity)",
    ).tag(config=True)

    min_star_prominence = Integer(
        3,
        help="Minimal star prominence over the background in terms of "
        "NSB variance std deviations",
    ).tag(config=True)

    max_star_magnitude = Float(
        7.0, help="Maximal magnitude of the star to be considered in the " "analysis"
    ).tag(config=True)

    cleaning = Dict(
        {"bound_thresh": 750, "pic_thresh": 15000}, help="Image cleaning parameters"
    ).tag(config=True)

    meteo_parameters = Dict(
        {"relative_humidity": 0.5, "temperature": 10, "pressure": 790},
        help="Meteorological parameters in  [dimensionless, deg C, hPa]",
    ).tag(config=True)

    psf_model_type = TelescopeParameter(
        trait=ComponentName(StatisticsExtractor, default_value="ComaModel"),
        default_value="ComaModel",
        help="Name of the PSFModel Subclass to be used.",
    ).tag(config=True)

    meteo_parameters = Dict(
        {"relative_humidity": 0.5, "temperature": 10, "pressure": 790},
        help="Meteorological parameters in  [dimensionless, deg C, hPa]",
    ).tag(config=True)

    def __init__(
        self,
        subarray,
        config=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(
            subarray=subarray,
            stats_extractor="Plain",
            config=config,
            parent=parent,
            **kwargs,
        )

        self.psf = PSFModel.from_name(
            self.psf_model_type, subarray=self.subarray, parent=self
        )

        self.location = EarthLocation(
            lon=self.telescope_location["longitude"] * u.deg,
            lat=self.telescope_location["latitude"] * u.deg,
            height=self.telescope_location["elevation"] * u.m,
        )

    def __call__(self, input_url, tel_id):
        self.tel_id = tel_id

        if self._check_req_data(input_url, "flatfield"):
            raise KeyError(
                "Relative gain not found. Gain calculation needs to be performed first."
            )

        # first get the camera geometry and pointing for the file and determine what stars we should see

        with EventSource(input_url, max_events=1) as src:
            self.camera_geometry = src.subarray.tel[self.tel_id].camera.geometry
            self.focal_length = src.subarray.tel[
                self.tel_id
            ].optics.equivalent_focal_length
            self.pixel_radius = self.camera_geometry.pixel_width[0]

            event = next(iter(src))

            self.pointing = SkyCoord(
                az=event.pointing.tel[self.telescope_id].azimuth,
                alt=event.pointing.tel[self.telescope_id].altitude,
                frame="altaz",
                obstime=event.trigger.time.utc,
                location=self.location,
            )  # get some pointing to make a list of stars that we expect to see

            self.pointing = self.pointing.transform_to("icrs")

            self.broken_pixels = np.unique(np.where(self.broken_pixels))

            self.image_size = len(
                event.variance_image.image
            )  # get the size of images of the camera we are calibrating

        self.stars_in_fov = Vizier.query_region(
            self.pointing, radius=Angle(2.0, "deg"), catalog="NOMAD"
        )[0]  # get all stars that could be in the fov

        self.stars_in_fov = self.stars_in_fov[
            self.tars_in_fov["Bmag"] < self.max_star_magnitude
        ]  # select stars for magnitude to exclude those we would not be able to see

        star_labels = [x.label for x in self.stars_in_fov]

        # get the accumulated variance images

        (
            accumulated_pointing,
            accumulated_times,
            variance_statistics,
        ) = self._get_accumulated_images(input_url)

        accumulated_images = np.array([x.mean for x in variance_statistics])

        star_pixels = self._get_expected_star_pixels(
            accumulated_times, accumulated_pointing
        )

        star_mask = np.ones(self.image_size, dtype=bool)

        star_mask[star_pixels] = False

        # get NSB values

        nsb = np.mean(accumulated_images[star_mask], axis=1)
        nsb_std = np.std(accumulated_images[star_mask], axis=1)

        clean_images = np.array([x - y for x, y in zip(accumulated_images, nsb)])

        reco_stars = []

        for i, image in enumerate(clean_images):
            reco_stars.append([])
            camera_frame = EngineeringCameraFrame(
                telescope_pointing=accumulated_pointing[i],
                focal_length=self.focal_length,
                obstime=accumulated_times[i].utc,
                location=self.location,
            )
            for star in self.stars_in_fov:
                reco_stars[-1].append(
                    self._fit_star_position(
                        star, accumulated_times[i], camera_frame, image, nsb_std[i]
                    )
                )

        return reco_stars

        # now fit the pointing correction.
        fitter = Pointing_Fitter(
            star_labels,
            self.pointing,
            self.location,
            self.focal_length,
            self.observed_wavelength,
            self.meteo_parameters,
        )

        fitter.fit(accumulated_pointing)

    def _check_req_data(self, url, calibration_type):
        """
        Check if the prerequisite calibration data exists in the files

        Parameters
        ----------
        url : str
            URL of file that is to be tested
        tel_id : int
            The telescope id.
        calibration_type : str
            Name of the field that is to be looked for e.g. flatfield or
            gain
        """
        with EventSource(url, max_events=1) as source:
            event = next(iter(source))

        calibration_data = getattr(event.mon.tel[self.tel_id], calibration_type)

        if calibration_data is None:
            return False

        return True

    def _calibrate_var_images(self, var_images, time, calibration_file):
        """
        Calibrate a set of variance images

        Parameters
        ----------
        var_images : list
            list of variance images
        time : list
            list of times correxponding to the variance images
        calibration_file : str
            name of the file where the calibration data can be found
        """
        # So i need to use the interpolator classes to read the calibration data
        relative_gains = FlatFieldInterpolator(
            calibration_file
        )  # this assumes gain_file is an hdf5 file. The interpolator will automatically search for the data in the default location when i call the instance

        for i, var_image in enumerate(var_images):
            var_images[i].image = np.divide(
                var_image.image,
                np.square(relative_gains(time[i])),
            )

        return var_images

    def _get_expected_star_pixels(self, time_list, pointing_list):
        """
        Determine which in which pixels stars are expected for a series of images

        Parameters
        ----------
        time_list : list
            list of time values where the images were captured
        pointing_list : list
            list of pointing values for the images
        """

        res = []

        for pointing, time in zip(
            pointing_list, time_list
        ):  # loop over time and pointing of images
            temp = []

            camera_frame = EngineeringCameraFrame(
                telescope_pointing=pointing,
                focal_length=self.focal_length,
                obstime=time.utc,
                location=self.location,
            )  # get the engineering camera frame for the pointing

            for star in self.stars_in_fov:
                star_coords = SkyCoord(
                    star["RAJ2000"], star["DEJ2000"], unit="deg", frame="icrs"
                )
                star_coords = star_coords.transform_to(camera_frame)
                expected_central_pixel = self.camera_geometry.transform_to(
                    camera_frame
                ).position_to_pix_index(
                    star_coords.x, star_coords.y
                )  # get where the star should be
                cluster = copy.deepcopy(
                    self.camera_geometry.neighbors[expected_central_pixel]
                )  # get the neighborhood of the star
                cluster_corona = []

                for pixel_index in cluster:
                    cluster_corona.extend(
                        copy.deepcopy(self.camera_geometry.neighbors[pixel_index])
                    )  # and add another layer of pixels to be sure

                cluster.extend(cluster_corona)
                cluster.append(expected_central_pixel)
                temp.extend(list(set(cluster)))

            res.append(temp)

        return res

    def _fit_star_position(
        self, star, timestamp, camera_frame, image, nsb_std, current_pointing
    ):
        star_coords = SkyCoord(
            star["RAJ2000"], star["DEJ2000"], unit="deg", frame="icrs"
        )

        star_coords = star_coords.transform_to(camera_frame)

        rho, phi = cart2pol(star_coords.x.to_value(u.m), star_coords.y.to_value(u.m))

        if phi < 0:
            phi = phi + 2 * np.pi

        star_container = StarContainer(
            label=star["NOMAD1"],
            magnitude=star["Bmag"],
            expected_x=star_coords.x,
            expected_y=star_coords.y,
            expected_r=rho * u.m,
            expected_phi=phi * u.rad,
            timestamp=timestamp,
        )

        current_geometry = self.camera_geometry.transform_to(camera_frame)

        hit_pdf = self._get_star_pdf(star, current_geometry)
        cluster = np.where(hit_pdf > self.pdf_percentile_limit * np.sum(hit_pdf))

        if not np.any(image[cluster] > self.min_star_prominence * nsb_std):
            self.log.info("Star %s can not be detected", star["NOMAD1"])
            star.pixels = np.full(self.max_cluster_size, -1)
            return star_container

        pad_size = self.max_cluster_size - len(cluster[0])
        if pad_size > 0:
            star.pixels = np.pad(cluster[0], (0, pad_size), constant_values=-1)
        else:
            star.pixels = cluster[0][: self.max_cluster_size]

            self.log.warning(
                "Reconstructed cluster is longer than %s, truncated cluster info will "
                "be recorded to the output table. Not a big deal, as correct cluster "
                "used for position reconstruction.",
                self.max_cluster_size,
            )
            return star_container

        rs, fs = cart2pol(
            current_geometry.pix_x[cluster].to_value(u.m),
            current_geometry.pix_y[cluster].to_value(u.m),
        )

        k, r0, sr = self.psf_model.radial_pdf_params

        star_container.reco_r = (
            self.coma_r_shift_correction
            * np.average(rs, axis=None, weights=image[cluster], returned=False)
            * u.m
        )

        star_container.reco_x = self.coma_r_shift_correction * np.average(
            current_geometry.pix_x[cluster],
            axis=None,
            weights=image[cluster],
            returned=False,
        )

        star_container.reco_y = self.coma_r_shift_correction * np.average(
            current_geometry.pix_y[cluster],
            axis=None,
            weights=image[cluster],
            returned=False,
        )

        _, star_container.reco_phi = cart2pol(star.reco_x, star.reco_y)

        if star_container.reco_phi < 0:
            star_container.reco_phi = star.reco_phi + 2 * np.pi * u.rad

        star_container.reco_dx = (
            np.sqrt(np.cov(current_geometry.pix_x[cluster], aweights=hit_pdf[cluster]))
            * u.m
        )

        star_container.reco_dy = (
            np.sqrt(np.cov(current_geometry.pix_y[cluster], aweights=hit_pdf[cluster]))
            * u.m
        )

        star_container.reco_dr = np.sqrt(np.cov(rs, aweights=hit_pdf[cluster])) * u.m

        _, star_container.reco_dphi = cart2pol(
            star_container.reco_dx, star_container.reco_dy
        )

        return star_container

    def _get_accumulated_images(self, input_url):
        # Read the whole dl1-like images of pedestal and flat-field data with the TableLoader
        input_data = TableLoader(input_url=input_url)
        dl1_table = input_data.read_telescope_events_by_id(
            telescopes=self.tel_id,
            dl1_images=True,
            dl1_parameters=False,
            dl1_muons=False,
            dl2=False,
            simulated=False,
            true_images=False,
            true_parameters=False,
            instrument=False,
            pointing=True,
        )

        # get the trigger type for all images and make a mask

        event_mask = dl1_table["event_type"] == 2

        # get the pointing for all images and filter for trigger type

        altitude = dl1_table["telescope_pointing_altitude"][event_mask]
        azimuth = dl1_table["telescope_pointing_azimuth"][event_mask]
        time = dl1_table["time"][event_mask]

        pointing = [
            SkyCoord(
                az=x, alt=y, frame="altaz", obstime=z.tai.utc, location=self.location
            )
            for x, y, z in zip(azimuth, altitude, time)
        ]

        # get the time and images from the data

        variance_images = copy.deepcopy(dl1_table["variance_image"][event_mask])

        # now make a filter to to reject EAS light and starlight and keep a separate EAS filter

        charge_images = dl1_table["image"][event_mask]

        light_mask = [
            tailcuts_clean(
                self.camera_geometry,
                x,
                picture_thresh=self.cleaning["pic_thresh"],
                boundary_thresh=self.cleaning["bound_thresh"],
            )
            for x in charge_images
        ]

        shower_mask = copy.deepcopy(light_mask)

        star_pixels = self._get_expected_star_pixels(time, pointing)

        light_mask[:, star_pixels] = True

        if self.broken_pixels is not None:
            light_mask[:, self.broken_pixels] = True

        # calculate the average variance in viable pixels and replace the values where there is EAS light

        mean_variance = np.mean(variance_images[~light_mask])

        variance_images[shower_mask] = mean_variance

        # now calibrate the images

        variance_images = self._calibrate_var_images(
            self, variance_images, time, input_url
        )

        # Get the average variance across the data to

        # then turn it into a table that the extractor can read
        variance_image_table = QTable([time, variance_images], names=["time", "image"])

        # Get the extractor
        extractor = self.stats_extractor[self.stats_extractor_type.tel[self.tel_id]]

        # get the cumulative variance images using the statistics extractor and return the value

        variance_statistics = extractor(variance_image_table)

        accumulated_times = np.array([x.validity_start for x in variance_statistics])

        # calculate where stars might be

        accumulated_pointing = np.array(
            [x for x in pointing if pointing.time in accumulated_times]
        )

        return (accumulated_pointing, accumulated_times, variance_statistics)

    def _get_star_pdf(self, star, current_geometry):
        image = np.zeros(self.image_size)

        r0 = star.expected_r.to_value(u.m)
        f0 = star.expected_phi.to_value(u.rad)

        self.psf_model.update_model_parameters(r0, f0)

        dr = (
            self.pdf_bin_size
            * np.rad2deg(np.arctan(1 / self.focal_length.to_value(u.m)))
            / 3600.0
        )
        r = np.linspace(
            r0 - dr * self.n_pdf_bins / 2.0,
            r0 + dr * self.n_pdf_bins / 2.0,
            self.n_pdf_bins,
        )
        df = np.deg2rad(self.pdf_bin_size / 3600.0) * 100
        f = np.linspace(
            f0 - df * self.n_pdf_bins / 2.0,
            f0 + df * self.n_pdf_bins / 2.0,
            self.n_pdf_bins,
        )

        for r_ in r:
            for f_ in f:
                val = self.psf_model.pdf(r_, f_) * dr * df
                x, y = pol2cart(r_, f_)
                pixelN = current_geometry.position_to_pix_index(x * u.m, y * u.m)
                if pixelN != -1:
                    image[pixelN] += val

        return image


class Pointing_Fitter:
    """
    Pointing correction fitter
    """

    def __init__(
        self,
        stars,
        times,
        telescope_pointing,
        telescope_location,
        focal_length,
        observed_wavelength,
        meteo_params,
        fit_grid="polar",
    ):
        """
        Constructor

        :param list stars: List of lists of star containers the first dimension is supposed to be time
        :param list time: List of time values for the when the star locations were fitted
        :param SkyCoord telescope_pointing: Telescope pointing in ICRS frame
        :param EarthLocation telescope_location: Telescope location
        :param Quantity[u.m] telescope_focal_length: Telescope focal length [m]
        :param Quantity[u.micron] observed_wavelength: Telescope focal length [micron]
        :param float relative_humidity: Relative humidity
        :param Quantity[u.deg_C] temperature: Temperature [C]
        :param Quantity[u.hPa] pressure: Pressure [hPa]
        :param str fit_grid: Coordinate system grid to use. Either polar or cartesian
        """
        self.star_containers = stars
        self.times = times
        self.telescope_pointing = telescope_pointing
        self.telescope_location = telescope_location
        self.focal_length = focal_length
        self.obswl = observed_wavelength
        self.relative_humidity = meteo_params["relative_humidity"]
        self.temperature = meteo_params["temperature"]
        self.pressure = meteo_params["pressure"]
        self.stars = []
        self.visible = []
        self.data = []
        self.errors = []
        # Construct the data here. Stars that were not found are marked in the variable "visible" and use the coordinates (0,0) whenever they can not be seen
        for star_list in stars:
            self.data.append([])
            self.errors.append([])
            self.visible.append({})
            for star in star_list:
                if star.reco_x != np.nan * u.m:
                    self.visible[-1].update({star.label: True})
                    self.data[-1].append(star.reco_x)
                    self.data[-1].append(star.reco_y)
                    self.errors[-1].append(star.reco_dx)
                    self.errors[-1].append(star.reco_dy)
                else:
                    self.visible[-1].update({star.label: False})
                    self.data[-1].append(0)
                    self.data[-1].append(0)
                    self.errors[-1].append(
                        1000.0
                    )  # large error value to suppress the stars that were not found
                    self.errors[-1].append(1000.0)

        for star in stars[0]:
            self.stars.append(self.init_star(star.label))
        self.fit_mode = "xy"
        self.fit_grid = fit_grid
        self.star_motion_model = Model(self.fit_function)
        self.fit_summary = None
        self.fit_resuts = None

    def init_star(self, star_label):
        """
        Initialize StarTracker object for a given star

        :param str star_label: Star label according to NOMAD catalog

        :return: StarTracker object
        """
        star = Vizier(catalog="NOMAD").query_constraints(NOMAD1=star_label)[0]
        star_coords = SkyCoord(
            star["RAJ2000"],
            star["DEJ2000"],
            unit="deg",
            frame="icrs",
            obswl=self.obswl,
            relative_humidity=self.relative_humidity,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        st = StarTracker(
            star_label,
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
        index = self.times.index(t)

        return self.telescope_pointing[index]

    def fit_function(self, p, t):
        """
        Construct the fit function for the pointing correction

        p: Fit parameters
        t: Timestamp in UNIX_TAI format

        """

        time = Time(t, format="unix_tai", scale="utc")
        index = self.times.index(time)
        coord_list = []

        m_ra, m_dec = p
        new_ra = self.current_pointing(time).ra + m_ra * u.deg
        new_dec = self.current_pointing(time).dec + m_dec * u.deg

        new_pointing = SkyCoord(ICRS(ra=new_ra, dec=new_dec))

        m_ra, m_dec = p
        new_ra = self.current_pointing(time).ra + m_ra * u.deg
        new_dec = self.current_pointing(time).dec + m_dec * u.deg
        new_pointing = SkyCoord(ICRS(ra=new_ra, dec=new_dec))
        for star in self.stars:
            if self.visible[index][star.label]:  # if star was visible give the value
                x, y = star.position_in_camera_frame(time, new_pointing)
            else:  # otherwise set it to (0,0) and set a large error value
                x, y = (0, 0)
            if self.fit_grid == "polar":
                x, y = cart2pol(x, y)
            coord_list.extend([x])
            coord_list.extend([y])

        return coord_list

    def fit(self, data, errors, time_range, fit_mode="xy"):
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
        if errors is not None:
            rdata = RealData(x=self.times, y=data, sy=errors)
        else:
            rdata = RealData(x=self.times, y=data)
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
