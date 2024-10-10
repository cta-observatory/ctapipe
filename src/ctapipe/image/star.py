import copy

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord

from ctapipe.containers import StarContainer
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.coordinates.utils import cart2pol, pol2cart
from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import (
    Dict,
    Float,
    Integer,
)
from ctapipe.image import tailcuts_clean
from ctapipe.image.extractor import VarianceExtractor
from ctapipe.image.psf_model import PSFModel
from ctapipe.utils.astro import get_bright_stars


def get_expected_star_pixels(stars, camera_frame, geometry):
    res = []

    for star in stars:
        star_coords = star["ra_dec"].transform_to(camera_frame)
        guessed_central_pixel = geometry.transform_to(
            camera_frame
        ).position_to_pix_index(star_coords.x, star_coords.y)
        res.extend(guessed_central_pixel)

    return list(set(res))


def get_star_pixel_mask(stars, camera_frame, geometry):
    mask = np.full(len(geometry), False)
    star_pixels = get_expected_star_pixels(stars, camera_frame, geometry)
    all_pixels = []
    for pixel in star_pixels:
        cluster = copy.deepcopy(geometry.neighbors[pixel])
        cluster_corona = []
        for pixel_index in cluster:
            cluster_corona.extend(copy.deepcopy(geometry.neighbors[pixel_index]))
        cluster.extend(cluster_corona)
        cluster.append(pixel)
        all_pixels.extend(cluster)

    mask[list(set(all_pixels))] = True

    return mask


class VarianceImageProducer(TelescopeComponent):

    """
    Produces calibrated and cleaned star images from SkyField events


    """

    tel_id = Integer(1, help="Telescope ID").tag(config=True)

    telescope_location = Dict(
        {"longitude": 342.108612, "latitude": 28.761389, "elevation": 2147},
        help="Telescope location, longitude and latitude should be expressed in deg, "
        "elevation - in meters",
    ).tag(config=True)

    lookup_radius = Float(2.0, help="Radius in degrees used to look up stars").tag(
        config=True
    )

    cleaning = Dict(
        {"bound_thresh": 750, "pic_thresh": 15000}, help="Image cleaning parameters"
    ).tag(config=True)

    telescope_location = Dict(
        {"longitude": 342.108612, "latitude": 28.761389, "elevation": 2147},
        help="Telescope location, longitude and latitude should be expressed in deg, "
        "elevation - in meters",
    ).tag(config=True)
    max_star_magnitude = Float(
        7.0, help="Maximal magnitude of the star to be considered in the " "analysis"
    ).tag(config=True)

    def __init__(self, subarray, **kwargs):
        self.subarray = subarray
        self.focal_length = self.subarray.tel[
            self.tel_id
        ].optics.equivalent_focal_length
        self.camera_geometry = self.subarray.tel[self.tel_id].camera.geometry
        self.location = EarthLocation(
            lon=self.telescope_location["longitude"] * u.deg,
            lat=self.telescope_location["latitude"] * u.deg,
            height=self.telescope_location["elevation"] * u.m,
        )

    def __call__(self, event):
        extractor = VarianceExtractor()

        # get the pointing

        pointing = SkyCoord(
            alt=event.pointing.tel[self.tel_id].altitude,
            az=event.pointing.tel[self.tel_id].azimuth,
            frame=AltAz(),
        )

        # make a frame for the current pointing

        camera_frame = EngineeringCameraFrame(
            telescope_pointing=pointing,
            focal_length=self.focal_length,
            obstime=event.trigger.time.utc,
            location=self.location,
        )

        stars = get_bright_stars(
            pointing=pointing,
            radius=self.lookup_radius * u.deg,
            magnitude_cut=self.max_star_magnitude,
        )

        variance_image = extractor(
            event.r1.tel[self.tel_id].waveform, self.tel_id, None, None
        )

        # calibrate the image

        ff = event.monitoring.tel[self.tel_id].FlatField

        variance_image.image = np.divide(variance_image.image, np.square(ff.median))

        light_mask = tailcuts_clean(
            self.camera_geometry,
            event.dl1.tel[self.tel_id].image,
            picture_thresh=self.cleaning["pic_thresh"],
            boundary_thresh=self.cleaning["bound_thresh"],
        )

        shower_mask = copy.deepcopy(light_mask)

        star_mask = get_star_pixel_mask(stars, camera_frame, self.geometry)

        light_mask[star_mask] = True

        broken_pixels = np.where(event.mon.tel[self.tel_id].calibration.unusable_pixels)

        if broken_pixels is not None:
            light_mask[broken_pixels] = True

        mean_variance = np.mean(variance_image.image[:, ~light_mask])

        variance_image.image[shower_mask] = mean_variance

        return variance_image


class StarExtractor(TelescopeComponent):

    """
    Produces list of star containers for a StatisticsContainer filled with cleaned Variance images
    """

    PSFModel = Dict(
        {
            "model": "ComaModel",
            "parameters": {
                "asymmetry": [0.49244797, 9.23573115, 0.15216096],
                "radial_scale": [0.01409259, 0.02947208, 0.06000271, -0.02969355],
                "az_scale": [0.24271557, 7.5511501, 0.02037972],
            },
        },
        help="PSF model description",
    ).tag(config=True)
    telescope_location = Dict(
        {"longitude": 342.108612, "latitude": 28.761389, "elevation": 2147},
        help="Telescope location, longitude and latitude should be expressed in deg, "
        "elevation - in meters",
    ).tag(config=True)
    lookup_radius = Float(2.0, help="Radius in degrees used to look up stars").tag(
        config=True
    )
    max_star_magnitude = Float(
        7.0, help="Maximal magnitude of the star to be considered in the " "analysis"
    ).tag(config=True)

    def __init(self, subarray):
        self.psf_model = PSFModel.from_name(self.PSFModel["model"])
        self.psf_model.update_model_parameters(self.PSFModel["parameters"])
        self.location = EarthLocation(
            lon=self.telescope_location["longitude"] * u.deg,
            lat=self.telescope_location["latitude"] * u.deg,
            height=self.telescope_location["elevation"] * u.m,
        )
        self.subarray = subarray
        self.focal_length = self.subarray.tel[
            self.tel_id
        ].optics.equivalent_focal_length
        self.camera_geometry = self.subarray.tel[self.tel_id].camera.geometry

    def __call__(self, statistics, pointing, time):
        image = statistics["mean"]
        stars = get_bright_stars(
            pointing=pointing,
            radius=self.lookup_radius * u.deg,
            magnitude_cut=self.max_star_magnitude,
        )

        camera_frame = EngineeringCameraFrame(
            telescope_pointing=pointing,
            focal_length=self.focal_length,
            obstime=time.utc,
            location=self.location,
        )

        self.current_geometry = self.camera_geometry.transform_to(camera_frame)

        expected_star_pixels = get_expected_star_pixels(
            stars, camera_frame, self.camera_geometry
        )
        star_mask = np.full(len(self.geometry), True)
        star_mask[expected_star_pixels] = False
        nsb = np.mean(image[star_mask])
        nsb_std = np.std(image[star_mask])
        clean_image = image - nsb
        clean_image[clean_image < 0] = 0
        all_stars = []
        for star in stars:
            star_container = self.reco_star(
                clean_image, star, time, camera_frame, nsb_std
            )
            all_stars.append(star_container)

        return all_stars

    def reco_star(self, image, star, time, frame, nsb_std):
        star_coords = star["ra_dec"].transform_to(frame)
        rho, phi = cart2pol(star_coords.x.to_value(u.m), star_coords.y.to_value(u.m))
        if phi < 0:
            phi = phi + 2 * np.pi

        star = StarContainer(
            label=star["name"],
            magnitude=star["vmag"],
            expected_x=star_coords.x,
            expected_y=star_coords.y,
            expected_r=rho * u.m,
            expected_phi=phi * u.rad,
            timestamp=time.utc,
        )

        star_pdf = self.get_star_pdf_cluster(star, frame)

        cluster = np.where(star_pdf > self.pdf_percentile_limit * np.sum(star_pdf))

        star_detected = False
        if np.any(image[cluster] > self.min_star_prominence * nsb_std):
            star_detected = True
        if not star_detected:
            star.pixels = np.full(self.max_cluster_size, -1)
            return star

        pad_size = self.max_cluster_size - len(cluster[0])
        if pad_size > 0:
            # pad cluster to achieve same size array
            star.pixels = np.pad(cluster[0], (0, pad_size), constant_values=-1)
        else:
            # In this case written cluster would be truncated, but no error will be raised
            star.pixels = cluster[0][: self.max_cluster_size]

        # Check if star is affected by broken/turned off pixels
        if np.any(np.isin(cluster, self.broken_pixels)):
            return star

        rs, fs = cart2pol(
            self.current_geometry.pix_x[cluster].to_value(u.m),
            self.current_geometry.pix_y[cluster].to_value(u.m),
        )

        k, r0, sr = self.psf_model.radial_pdf_params

        star.reco_r = (
            self.coma_r_shift_correction
            * np.average(rs, axis=None, weights=image[cluster], returned=False)
            * u.m
        )

        star.reco_x = self.coma_r_shift_correction * np.average(
            self.current_geometry.pix_x[cluster],
            axis=None,
            weights=image[cluster],
            returned=False,
        )
        star.reco_y = self.coma_r_shift_correction * np.average(
            self.current_geometry.pix_y[cluster],
            axis=None,
            weights=image[cluster],
            returned=False,
        )
        _, star.reco_phi = cart2pol(star.reco_x, star.reco_y)
        if star.reco_phi < 0:
            star.reco_phi = star.reco_phi + 2 * np.pi * u.rad
        star.reco_dx = (
            np.sqrt(
                np.cov(self.current_geometry.pix_x[cluster], aweights=star_pdf[cluster])
            )
            * u.m
        )
        star.reco_dy = (
            np.sqrt(
                np.cov(self.current_geometry.pix_y[cluster], aweights=star_pdf[cluster])
            )
            * u.m
        )
        star.reco_dr = np.sqrt(np.cov(rs, aweights=star_pdf[cluster])) * u.m

        _, star.reco_dphi = cart2pol(star.reco_dx, star.reco_dy)

        return star

    def get_star_pdf_cluster(self, star, frame):
        image = np.zeros(len(self.geometry))

        r0 = star.expected_r.to_value(u.m)
        f0 = star.expected_phi.to_value(u.rad)
        self.psf_model.update_position(r0, f0)

        dr = (
            self.pdf_bin_size
            * np.rad2deg(np.arctan(1 / self.focal_length.to_value(u.m)))
            / 3600.0
        )  # dr in meters
        r = np.linspace(
            r0 - dr * self.n_pdf_bins / 2.0,
            r0 + dr * self.n_pdf_bins / 2.0,
            self.n_pdf_bins,
        )
        df = (
            np.deg2rad(self.pdf_bin_size / 3600.0) * 100
        )  # convert to radians, increase by x100
        f = np.linspace(
            f0 - df * self.n_pdf_bins / 2.0,
            f0 + df * self.n_pdf_bins / 2.0,
            self.n_pdf_bins,
        )
        for r_ in r:
            for f_ in f:
                self.log.debug("Calculate pdf for point")
                val = self.psf_model.pdf(r_, f_) * dr * df
                self.log.debug("PDF calculated")
                x, y = pol2cart(r_, f_)
                pixelN = self.current_geometry.position_to_pix_index(x * u.m, y * u.m)
                if pixelN != -1:  # Some of the hits fall in the inter-pixel blind spots
                    image[pixelN] += val
        return image
