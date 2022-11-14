from functools import partial

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from iminuit import Minuit

from ctapipe.containers import Model3DReconstructedGeometryContainer
from ctapipe.coordinates import CameraFrame, TelescopeFrame
from ctapipe.core.traits import Unicode
from ctapipe.image import (
    GaussianShowermodel,
    ShowermodelPredictor,
    neg_log_likelihood_approx,
)
from ctapipe.reco.reco_algorithms import Reconstructor


class Model3DGeometryReconstuctor(Reconstructor):
    geometry_seed = Unicode(default_value="HillasReconstructor").tag(config=True)

    def __init__(self, subarray, **kwargs):
        super().__init__(subarray=subarray, **kwargs)

    def __call__(self, event):
        if self.geometry_seed not in event.dl2.stereo.geometry:
            raise ValueError()

        # get all telescope properties that we need for the ShowermodelPredictor
        tel_positions = {}
        tel_solid_angles = {}
        tel_mirror_area = {}
        tel_spe_widths = {}
        tel_pedestial_widths = {}
        for tel_id in event.dl1.tel.keys():
            tel_positions[tel_id] = self.subarray.positions[tel_id].to_value(u.m)
            geometry = self.subarray.tel[tel_id].camera.geometry
            tel_solid_angles[tel_id] = geometry.transform_to(
                TelescopeFrame()
            ).pix_area.to_value(u.rad**2)
            tel_mirror_area[tel_id] = self.subarray.tel[
                tel_id
            ].optics.mirror_area.to_value(u.m**2)
            tel_spe_widths[tel_id] = (
                spe
                if (spe := event.mon.tel[tel_id].flatfield.charge_std) is not None
                else 0.5
            )
            tel_pedestial_widths[tel_id] = (
                ped
                if (ped := event.mon.tel[tel_id].pedestal.charge_std) is not None
                else 2.8
            )

        self.tel_positions = tel_positions
        self.tel_solid_angles = tel_solid_angles
        self.tel_mirror_area = tel_mirror_area
        self.tel_pix_coords_altaz = self._tel_pix_coords_altaz(event)

        self.predictor = ShowermodelPredictor(
            self.tel_positions,
            self.tel_pix_coords_altaz,
            self.tel_solid_angles,
            self.tel_mirror_area,
        )

        shower_parameters, errors = self._fit(event)

        event.dl2.stereo.geometry[
            self.__class__.__name__
        ] = Model3DReconstructedGeometryContainer(
            total_photons=shower_parameters["total_photons"],
            total_photons_uncert=errors["total_photons"],
            core_x=shower_parameters["x"],
            core_y=shower_parameters["y"],
            core_uncert_x=errors["x"],
            core_uncert_y=errors["y"],
            alt=shower_parameters["altitude"],
            az=shower_parameters["azimuth"],
            alt_uncert=errors["altitude"],
            az_uncert=errors["azimuth"],
            h_max=shower_parameters["h_max"],
            h_max_uncert=errors["h_max"],
            width=shower_parameters["width"],
            width_uncert=errors["width"],
            length=shower_parameters["length"],
            length_uncert=errors["length"],
            telescopes=list(event.dl1.tel.keys()),
            is_valid=True,
            prefix=self.__class__.__name__,
        )

    def _fit(self, event):
        seeds = self._seeds(event)

        minimizer = Minuit(
            partial(self._likelihood, dl1_cam_container=event.dl1.tel.items()),
            **seeds,
            name=seeds.keys(),
        )

        minimizer.limits["total_photons"] = [0, None]
        minimizer.limits["x"] = [None, None]
        minimizer.limits["y"] = [None, None]
        minimizer.limits["altitude"] = [0, 90]
        minimizer.limits["azimuth"] = [0, 360]
        minimizer.limits["h_max"] = [0, None]
        minimizer.limits["width"] = [1e-8, None]
        minimizer.limits["length"] = [1e-8, None]

        minimizer.errordef = Minuit.LIKELIHOOD
        minimizer.migrad()
        # from IPython import embed

        # embed()
        fit = minimizer.values
        fit_errors = minimizer.errors

        return fit, fit_errors

    def _likelihood(
        self,
        total_photons,
        x,
        y,
        azimuth,
        altitude,
        h_max,
        width,
        length,
        dl1_cam_container,
    ):

        model = GaussianShowermodel(
            total_photons,
            x,
            y,
            azimuth,
            altitude,
            h_max,
            width,
            length,
        )

        self.predictor.showermodel = model
        prediction = self.predictor.generate_images()

        log_likelihood = 0
        for tel_id, dl1 in dl1_cam_container:
            log_likelihood += np.sum(
                neg_log_likelihood_approx(dl1.image, prediction[tel_id], 0.5, 2.8)
            )

        return log_likelihood / len(dl1_cam_container)

    def _seeds(self, event):
        # get seeds from seed reconstructors 'HillasReconstructor'
        geometry_reconstructor = event.dl2.stereo.geometry[self.geometry_seed]
        total_photons = 100 * geometry_reconstructor.average_intensity
        az = geometry_reconstructor.az
        alt = geometry_reconstructor.alt
        x = geometry_reconstructor.core_x
        y = geometry_reconstructor.core_y
        width = 20 * u.m
        length = 3000 * u.m
        h_max = geometry_reconstructor.h_max

        return {
            "total_photons": total_photons,
            "x": x.to_value(u.m),
            "y": y.to_value(u.m),
            "azimuth": az.to_value(u.deg),
            "altitude": alt.to_value(u.deg),
            "h_max": h_max.to_value(u.m),
            "width": width.to_value(u.m),
            "length": length.to_value(u.m),
        }

    def _tel_pix_coords_altaz(self, event):
        tel_pix_coords_altaz = {}
        for tel_id in event.dl1.tel.keys():
            geometry = self.subarray.tel[tel_id].camera.geometry
            # (x,y)->(y,x) since this is also in a NorthingEasting frame instead of EastingNorthing similar to tel_positions
            pix_x = geometry.pix_y
            pix_y = geometry.pix_x
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
