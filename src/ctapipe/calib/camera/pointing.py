"""
Definition of the `CameraCalibrator` class, providing all steps needed to apply
calibration and image extraction, as well as supporting algorithms.
"""

import copy
from functools import cache

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.table import QTable
from astroquery.vizier import Vizier  # discuss this dependency with max etc.
from scipy.odr import ODR, RealData

from ctapipe.containers import StarContainer
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import (
    ComponentName,
    Dict,
    Float,
    Integer,
    TelescopeParameter,
    Unicode,
)
from ctapipe.image import tailcuts_clean
from ctapipe.image.psf_model import PSFModel
from ctapipe.instrument import CameraGeometry
from ctapipe.monitoring.aggregator import StatisticsAggregator
from ctapipe.monitoring.interpolation import FlatFieldInterpolator, PointingInterpolator

__all__ = ["PointingCalculator", "StarImageGenerator"]


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


def get_index_step(val, lookup):
    index = 0

    for i, x in enumerate(lookup):
        if val <= x:
            index = i

    if val > lookup[-1]:
        index = len(lookup) - 1

    return index


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


def get_star_pdf(r0, f0, geometry, psf, n_pdf_bins, pdf_bin_size, focal_length):
    image = np.zeros(len(geometry))

    psf.update_model_parameters(r0, f0)

    dr = pdf_bin_size * np.rad2deg(np.arctan(1 / focal_length)) / 3600.0
    r = np.linspace(
        r0 - dr * n_pdf_bins / 2.0,
        r0 + dr * n_pdf_bins / 2.0,
        n_pdf_bins,
    )
    df = np.deg2rad(pdf_bin_size / 3600.0) * 100
    f = np.linspace(
        f0 - df * n_pdf_bins / 2.0,
        f0 + df * n_pdf_bins / 2.0,
        n_pdf_bins,
    )

    for r_ in r:
        for f_ in f:
            val = psf.pdf(r_, f_) * dr * df
            x, y = pol2cart(r_, f_)
            pixelN = geometry.position_to_pix_index(x * u.m, y * u.m)
            if pixelN != -1:
                image[pixelN] += val

    return image


def StarImageGenerator(
    self,
    radius,
    phi,
    magnitude,
    n_pdf_bins,
    pdf_bin_size,
    psf_model_name,
    psf_model_pars,
    camera_name,
    focal_length,
):
    """
    :param list stars: list of star containers, stars to be placed in image
    :param dict psf_model_pars: psf model parameters
    """
    camera_geometry = CameraGeometry.from_name(camera_name)
    psf = PSFModel.from_name(self.psf_model_type, subarray=self.subarray, parent=self)
    psf.update_model_parameters(psf_model_pars)
    image = np.zeros(len(camera_geometry))
    for r, p, m in zip(radius, phi, magnitude):
        image += m * get_star_pdf(
            r, p, camera_geometry, psf, n_pdf_bins, pdf_bin_size, focal_length
        )

    return image


class StarTracer:
    """
    Utility class to trace a set of stars over a period of time and generate their locations in the camera
    """

    def __init__(
        self,
        stars,
        magnitude,
        az,
        alt,
        time,
        meteo_parameters,
        observed_wavelength,
        camera_geometry,
        focal_length,
        location,
    ):
        """
        param dict stars: dict of Astropy.SkyCoord objects, keys are the nomad labels
        param dict magnitude:
        param list time: list of Astropy.time objects corresponding to the altitude and azimuth values
        """

        self.stars = stars
        self.magnitude = magnitude

        self.pointing = PointingInterpolator()
        pointing = QTable([az, alt, time], names=["azimuth", "altitude", "time"])
        self.pointing.add_table(0, pointing)

        self.meteo_parameters = meteo_parameters
        self.observed_wavelength = observed_wavelength
        self.camera_geometry = camera_geometry
        self.focal_length = focal_length * u.m
        self.location = location

    @classmethod
    def from_lookup(
        cls,
        max_star_magnitude,
        az,
        alt,
        time,
        meteo_params,
        observed_wavelength,
        camera_geometry,
        focal_length,
        location,
    ):
        """
        classmethod to use vizier lookup to generate a class instance
        """
        _pointing = SkyCoord(
            az=az[0],
            alt=alt[0],
            frame="altaz",
            obstime=time[0],
            location=location,
            obswl=observed_wavelength * u.micron,
            relative_humidity=meteo_params["relative_humidity"],
            temperature=meteo_params["temperature"] * u.deg_C,
            pressure=meteo_params["pressure"] * u.hPa,
        )
        stars_in_fov = Vizier.query_region(
            _pointing, radius=Angle(2.0, "deg"), catalog="NOMAD"
        )[0]  # get all stars that could be in the fov

        stars_in_fov = stars_in_fov[stars_in_fov["Bmag"] < max_star_magnitude]

        stars = {}
        magnitude = {}
        for star in stars_in_fov:
            star_coords = {
                star["NOMAD1"]: SkyCoord(
                    star["RAJ2000"], star["DEJ2000"], unit="deg", frame="icrs"
                )
            }
            stars.update(star_coords)
            magnitude.update({star["NOMAD1"]: star["Bmag"]})

        return cls(
            stars,
            magnitude,
            az,
            alt,
            time,
            meteo_params,
            observed_wavelength,
            camera_geometry,
            focal_length,
            location,
        )

    def get_star_labels(self):
        """
        Return a list of all stars that are being traced
        """

        return list(self.stars.keys())

    def get_magnitude(self, star):
        """
        Return the magnitude of star

        parameter str star: NOMAD1 label of the star
        """

        return self.magnitude[star]

    def get_pointing(self, t, offset=(0.0, 0.0)):
        alt, az = self.pointing(t)
        alt += offset[0] * u.rad
        az += offset[1] * u.rad

        coords = SkyCoord(
            az=az,
            alt=alt,
            frame="altaz",
            obstime=t,
            location=self.location,
            obswl=self.observed_wavelength * u.micron,
            relative_humidity=self.meteo_parameters["relative_humidity"],
            temperature=self.meteo_parameters["temperature"] * u.deg_C,
            pressure=self.meteo_parameters["pressure"] * u.hPa,
        )

        return coords

    def get_camera_frame(self, t, offset=(0.0, 0.0), focal_correction=0.0):
        altaz_pointing = self.get_pointing(t, offset=offset)

        camera_frame = EngineeringCameraFrame(
            telescope_pointing=altaz_pointing,
            focal_length=self.focal_length + focal_correction * u.m,
            obstime=t,
            location=self.location,
        )

        return camera_frame

    def get_current_geometry(self, t, offset=(0.0, 0.0), focal_correction=0.0):
        camera_frame = self.get_camera_frame(
            t, offset=offset, focal_correction=focal_correction
        )

        current_geometry = self.camera_geometry.transform_to(camera_frame)

        return current_geometry

    def get_position_in_camera(self, t, star, offset=(0.0, 0.0), focal_correction=0.0):
        camera_frame = self.get_camera_frame(
            t, offset=offset, focal_correction=focal_correction
        )
        # Calculate the stars coordinates in the current camera frame
        coords = self.stars[star].transform_to(camera_frame)
        return (coords.x.to_value(), coords.y.to_value())

    def get_position_in_pixel(self, t, star, focal_correction=0.0):
        x, y = self.get_position_in_camera(t, star, focal_correction=focal_correction)
        current_geometry = self.get_current_geometry(t)

        return current_geometry.position_to_pix_index(x, y)

    def get_expected_star_pixels(self, t, focal_correction=0.0):
        """
        Determine which in which pixels stars are expected for a series of images

        Parameters
        ----------
        t : list
            list of time values where the images were captured
        """

        res = []

        for star in self.get_star_labels():
            expected_central_pixel = self.get_positions_in_pixel(
                t, star, focal_correction=focal_correction
            )
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
            res.extend(list(set(cluster)))

        return res


class PointingCalculator(TelescopeComponent):
    """
    Component to calculate pointing corrections from interleaved skyfield events.

    Attributes
    ----------
    stats_aggregator: str
        The name of the StatisticsAggregator subclass to be used to calculate the statistics of an image set
    telescope_location: dict
        The location of the telescope for which the pointing correction is to be calculated
    """

    stats_aggregator = TelescopeParameter(
        trait=ComponentName(StatisticsAggregator, default_value="PlainAggregator"),
        default_value="PlainAggregator",
        help="Name of the StatisticsAggregator Subclass to be used.",
    ).tag(config=True)

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
        7.0, help="Maximal magnitude of the star to be considered in the analysis"
    ).tag(config=True)

    cleaning = Dict(
        {"bound_thresh": 750, "pic_thresh": 15000}, help="Image cleaning parameters"
    ).tag(config=True)

    meteo_parameters = Dict(
        {"relative_humidity": 0.5, "temperature": 10, "pressure": 790},
        help="Meteorological parameters in  [dimensionless, deg C, hPa]",
    ).tag(config=True)

    psf_model_type = Unicode(
        "ComaModel", help="Name of the PSFModel Subclass to be used."
    ).tag(config=True)

    meteo_parameters = Dict(
        {"relative_humidity": 0.5, "temperature": 10, "pressure": 790},
        help="Meteorological parameters in  [dimensionless, deg C, hPa]",
    ).tag(config=True)

    n_pdf_bins = Integer(1000, help="Camera focal length").tag(config=True)

    pdf_bin_size = Float(10.0, help="Camera focal length").tag(config=True)

    focal_length = Float(1.0, help="Camera focal length in meters").tag(config=True)

    def __init__(
        self,
        subarray,
        geometry,
        config=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(
            subarray=subarray,
            stats_extractor="Plain",
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

        self.image_aggregator = StatisticsAggregator.from_name(
            self.stats_aggregator, subarray=self.subarray, parent=self
        )

        self.set_camera(geometry)

    def set_camera(self, geometry, focal_lengh):
        if isinstance(geometry, str):
            self.camera_geometry = CameraGeometry.from_name(geometry)

        self.pixel_radius = self.camera_geometry.pixel_width[0]

    def ingest_data(self, data_table):
        """
        Attributes
        ----------
        data_table : Table
            Table containing a series of variance images with corresponding initial pointing values, trigger times and calibration data
        """

        # set up the StarTracer here to track stars in the camera
        self.tracer = StarTracer.from_lookup(
            data_table["telescope_pointing_azimuth"],
            data_table["telescope_pointing_altitude"],
            data_table["time"],
            self.meteo_parameters,
            self.observed_wavelength,
            self.camera_geometry,
            self.focal_length,
            self.telescope_location,
        )

        self.broken_pixels = np.unique(np.where(data_table["unusable_pixels"]))

        for azimuth, altitude, time in zip(
            data_table["telescope_pointing_azimuth"],
            data_table["telescope_pointing_altitude"],
            data_table["time"],
        ):
            _pointing = SkyCoord(
                az=azimuth,
                alt=altitude,
                frame="altaz",
                obstime=time,
                location=self.location,
                obswl=self.observed_wavelength * u.micron,
                relative_humidity=self.meteo_parameters["relative_humidity"],
                temperature=self.meteo_parameters["temperature"] * u.deg_C,
                pressure=self.meteo_parameters["pressure"] * u.hPa,
            )
            self.pointing.append(_pointing)

        self.image_size = len(
            data_table["variance_images"][0].image
        )  # get the size of images of the camera we are calibrating

        star_labels = [x.label for x in self.stars_in_fov]

        # get the accumulated variance images

        self._get_accumulated_images(data_table)

    def fit_stars(self):
        stars = self.tracer.get_star_labels()

        self.all_containers = []

        for t, image in (self.accumulated_times, self.accumulated_images):
            self.all_containers.append([])

            for star in stars:
                container = self._fit_star_position(star, t, image, self.nsb_std)

                self.all_containers[-1].append(container)

        return self.all_containers

    def _fit_star_position(self, star, t, image, nsb_std):
        x, y = self.tracer.get_position_in_camera(self, t, star)

        rho, phi = cart2pol(x, y)

        if phi < 0:
            phi = phi + 2 * np.pi

        star_container = StarContainer(
            label=star,
            magnitude=self.tracer.get_magnitude(star),
            expected_x=x,
            expected_y=y,
            expected_r=rho * u.m,
            expected_phi=phi * u.rad,
            timestamp=t,
        )

        current_geometry = self.tracer.get_current_geometry(t)

        hit_pdf = get_star_pdf(
            rho,
            phi,
            current_geometry,
            self.psf,
            self.n_pdf_bins,
            self.pdf_bin_size,
            self.focal_length.to_value(u.m),
        )
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

    def _get_accumulated_images(self, data_table):
        variance_images = data_table["variance_images"]

        # now make a filter to to reject EAS light and starlight and keep a separate EAS filter

        light_mask = [
            tailcuts_clean(
                self.camera_geometry,
                x,
                picture_thresh=self.cleaning["pic_thresh"],
                boundary_thresh=self.cleaning["bound_thresh"],
            )

            for x in data_table["charge_image"]
        ]

        shower_mask = copy.deepcopy(light_mask)

        star_pixels = [
            self.tracer.get_expected_star_pixels(t) for t in data_table["time"]
        ]

        light_mask[:, star_pixels] = True

        if self.broken_pixels is not None:
            light_mask[:, self.broken_pixels] = True

        # calculate the average variance in viable pixels and replace the values where there is EAS light

        mean_variance = np.mean(variance_images[~light_mask])

        variance_images[shower_mask] = mean_variance

        # now calibrate the images

        relative_gains = FlatFieldInterpolator()
        relative_gains.add_table(0, data_table["relative_gain"])

        for i, var_image in enumerate(variance_images):
            variance_images[i] = np.divide(
                var_image,
                np.square(relative_gains(data_table["time"][i]).median),
            )

        # then turn it into a table that the aggregator can read
        variance_image_table = QTable(
            [data_table["time"], variance_images], names=["time", "image"]
        )

        # get the cumulative variance images using the statistics aggregator and return the value

        variance_statistics = self.image_aggregator(
            variance_image_table, col_name="image"
        )

        self.accumulated_times = np.array(
            [x.validity_start for x in variance_statistics]
        )

        accumulated_images = np.array([x.mean for x in variance_statistics])

        star_pixels = [
            self.tracer.get_expected_star_pixels(t) for t in data_table["time"]
        ]

        star_mask = np.ones(self.image_size, dtype=bool)

        star_mask[star_pixels] = False

        # get NSB values

        nsb = np.mean(accumulated_images[star_mask], axis=1)
        self.nsb_std = np.std(accumulated_images[star_mask], axis=1)

        self.clean_images = np.array([x - y for x, y in zip(accumulated_images, nsb)])

    def fit_function(self, p, t):
        """
        Construct the fit function for the pointing correction

        p: Fit parameters
        t: Timestamp in UNIX_TAI format

        """

        coord_list = []

        index = get_index_step(
            t, self.accumulated_times
        )  # this gives you the index corresponding to the
        for star in self.all_containers[index]:
            if not np.isnan(star.reco_x):
                x, y = self.tracer.get_position_in_camera(star.label, t, offset=p)
                coord_list.extend([x])
                coord_list.extend([y])

        return coord_list

    def fit(self):
        """
        Performs the ODR fit of stars trajectories and saves the results as self.fit_results

        :param str fit_mode: Fit mode. Can be 'y', 'xy' (default), 'xyz' or 'radec'.
        """

        results = []
        for i, t in enumerate(self.accumulated_times):
            init_mispointing = [0, 0]
            data = []
            errors = []
            for star in self.all_containers:
                if not np.isnan(star.reco_x):
                    data.append(star.reco_x)
                    data.append(star.reco_y)
                    errors.append(star.reco_dx)
                    errors.append(star.reco_dy)

            rdata = RealData(x=[t], y=data, sy=self.errors)
            odr = ODR(rdata, self.fit_function, beta0=init_mispointing)
            fit_summary = odr.run()
            fit_results = pd.DataFrame(
                data={
                    "dAZ": [fit_summary.beta[0]],
                    "dALT": [fit_summary.beta[1]],
                    "eAZ": [fit_summary.sd_beta[0]],
                    "eALT": [fit_summary.sd_beta[1]],
                }
            )
            results.append(fit_results)
        return results
