import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from iminuit import Minuit
from scipy.stats import poisson

from ctapipe.containers import ReconstructedGeometryContainer
from ctapipe.coordinates import CameraFrame, TelescopeFrame
from ctapipe.core.traits import Unicode
from ctapipe.image import GaussianShowermodel, ShowermodelPredictor
from ctapipe.reco import Reconstructor


class ShowerReconstructor(Reconstructor):
    geometry_seed = Unicode(default_value="HillasReconstructor").tag(config=True)

    def __init__(self, subarray, **kwargs):
        super().__init__(subarray=subarray, **kwargs)

    def __call__(self, event):
        if self.geometry_seed not in event.dl2.stereo.geometry:
            raise ValueError()
        # fit shower parameters using likelihood
        self.event = event  # ugly, only for _likelihood

        # get all telescope properties that we need for the ShowermodelPredictor
        tel_positions = {}
        tel_solid_angles = {}
        tel_mirror_area = {}
        for tel_id in event.dl1.tel.keys():
            tel_positions[tel_id] = self.subarray.positions[tel_id]
            geometry = self.subarray.tel[tel_id].camera.geometry
            tel_solid_angles[tel_id] = geometry.transform_to(TelescopeFrame()).pix_area
            tel_mirror_area[tel_id] = self.subarray.tel[tel_id].optics.mirror_area

        self.tel_positions = tel_positions
        self.tel_solid_angles = tel_solid_angles
        self.tel_mirror_area = tel_mirror_area

        shower_parameters, errors = self._fit(event)
        # store results in event structure
        event.dl2.stereo.geometry[
            self.__class__.__name__
        ] = ReconstructedGeometryContainer(
            alt=shower_parameters["altitude"],
            az=shower_parameters["azimuth"],
            alt_uncert=errors["altitude"],
            az_uncert=errors["azimuth"],
            core_x=shower_parameters["x"],
            core_y=shower_parameters["y"],
            core_tilted_x=None,
            core_tilted_y=None,
            h_max=shower_parameters["first_interaction"]
            - shower_parameters["width"]
            / 2
            * np.cos(90 - shower_parameters["altitude"]),
            average_intensity=shower_parameters["total_photons"],
            telescopes=list(event.dl1.tel.keys()),
            is_valid=True,
            prefix=self.__class__.__name__,
        )

    def _fit(self, event):
        # just call iminiut with likelihood and some seeds
        # may also need limits
        seeds = self._seeds(event)
        minimizer = Minuit(self._likelihood, *seeds)
        minimizer.limits = [
            (0, None),
            (-1000, 1000),
            (-1000, 1000),
            (0, 360),
            (0, 90),
            (0, None),
            (0, None),
            (0, None),
        ]
        minimizer.migrad()
        fit = minimizer.values
        fit_errors = minimizer.errors

        return fit, fit_errors

    def _likelihood(
        self, total_photons, x, y, azimuth, altitude, first_interaction, width, length
    ):
        # this defines the likelihood

        # generate showermodel with `shower_parameters`
        model = GaussianShowermodel(
            total_photons,
            x * u.m,
            y * u.m,
            azimuth * u.deg,
            altitude * u.deg,
            first_interaction * u.m,
            width * u.m,
            length * u.m,
        )

        # predict shower images for each telescope in the event structure
        tel_pix_coords_altaz = self._tel_pix_coords_altaz(self.event)
        predictor = ShowermodelPredictor(
            self.tel_positions,
            tel_pix_coords_altaz,
            self.tel_solid_angles,
            self.tel_mirror_area,
            showermodel=model,
        )

        prediction = predictor.generate_images()
        # calaculate pixewise likelihood of the predicted images to the event images
        log_likelihood = 0
        for tel_id, DL1CamContainer in self.event.dl1.tel.items():
            log_likelihood += np.sum(
                poisson.logpmf(k=DL1CamContainer.image, mu=prediction[tel_id].value)
            )

        return -log_likelihood

    def _seeds(self, event):
        # get seeds from seed reconstructors 'HillasReconstructor'
        geometry_reconstructor = event.dl2.stereo.geometry[self.geometry_seed]
        total_photons = 10 * geometry_reconstructor.average_intensity
        az = geometry_reconstructor.az
        alt = geometry_reconstructor.alt
        x = geometry_reconstructor.core_x
        y = geometry_reconstructor.core_y
        width = 20 * u.m
        length = 3000 * u.m
        first_interaction = geometry_reconstructor.h_max + width / 2 * np.cos(
            90 * u.deg - alt
        )

        return (
            total_photons,
            x.to_value(u.m),
            y.to_value(u.m),
            az.to_value(u.deg),
            alt.to_value(u.deg),
            first_interaction.to_value(u.m),
            width.to_value(u.m),
            length.to_value(u.m),
        )

    def _tel_pix_coords_altaz(self, event):
        tel_pix_coords_altaz = {}
        for tel_id in event.dl1.tel.keys():
            geometry = self.subarray.tel[tel_id].camera.geometry
            pix_x = geometry.pix_x
            pix_y = geometry.pix_y
            focal_length = self.subarray.tel[tel_id].optics.equivalent_focal_length

            pointing = event.pointing.tel[tel_id]
            altaz = AltAz(az=pointing.azimuth, alt=pointing.altitude)
            camera_frame = CameraFrame(
                focal_length=focal_length, telescope_pointing=altaz
            )

            cam_coords = SkyCoord(x=pix_x, y=pix_y, frame=camera_frame)

            cam_altaz = cam_coords.transform_to(AltAz())
            tel_pix_coords_altaz[tel_id] = cam_altaz

        return tel_pix_coords_altaz
