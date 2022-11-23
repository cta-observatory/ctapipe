from functools import partial

import astropy.units as u
import numpy as np
from iminuit import Minuit

from ctapipe.containers import Model3DReconstructedGeometryContainer
from ctapipe.core.traits import FloatTelescopeParameter, Unicode
from ctapipe.image import (
    GaussianShowermodel,
    ShowermodelPredictor,
    neg_log_likelihood_approx,
)
from ctapipe.reco import Reconstructor


class Model3DGeometryReconstructor(Reconstructor):
    geometry_seed = Unicode(
        default_value="HillasReconstructor",
        help="GeometryReconstructor that is used as a seed for the likelihood fit",
    ).tag(config=True)

    pedestal_charge_std = FloatTelescopeParameter(
        default_value=1.4,
        help="Pedestal charge std",
    ).tag(config=True)

    flatfield_charge_std = FloatTelescopeParameter(
        default_value=0.6, help="Flatfield charge std"
    ).tag(config=True)

    def __init__(self, subarray, **kwargs):
        super().__init__(subarray=subarray, **kwargs)
        self.predictor = ShowermodelPredictor(self.subarray)

    def __call__(self, event):
        if self.geometry_seed not in event.dl2.stereo.geometry:
            raise KeyError(
                f"The geometry seed {self.geometry_seed} could not be provided. Run the reconstructors in the correct order."
            )

        tel_spe_widths = {}
        tel_pedestal_widths = {}
        for tel_id in event.dl1.tel.keys():
            tel_spe_widths[tel_id] = (
                spe
                if (spe := event.mon.tel[tel_id].flatfield.charge_std) is not None
                else self.flatfield_charge_std.tel[tel_id]
            )
            tel_pedestal_widths[tel_id] = (
                ped
                if (ped := event.mon.tel[tel_id].pedestal.charge_std) is not None
                else self.pedestal_charge_std.tel[tel_id]
            )

        self.tel_spe_widths = tel_spe_widths
        self.tel_pedestal_widths = tel_pedestal_widths

        self.predictor.pointing(event)
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

        images = {}
        for tel_id in event.dl1.tel.keys():
            image = event.dl1.tel[tel_id].image.copy()
            images[tel_id] = image

        minimizer = Minuit(
            partial(self._likelihood, images=images),
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
        images,
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
        for tel_id, img in images.items():
            log_likelihood += np.sum(
                neg_log_likelihood_approx(
                    img,
                    prediction[tel_id],
                    self.tel_spe_widths[tel_id],
                    self.tel_pedestal_widths[tel_id],
                )
            )

        return log_likelihood / len(images)

    def _seeds(self, event):
        # get seeds from seed reconstructors 'HillasReconstructor'
        geometry_reconstructor = event.dl2.stereo.geometry[self.geometry_seed]
        total_photons = (
            100 * average_intensity
            if not np.isnan(
                average_intensity := geometry_reconstructor.average_intensity
            )
            else 1e6
        )
        az = (
            az
            if not np.isnan(az := geometry_reconstructor.az)
            else event.pointing.array_azimuth
        )
        alt = (
            alt
            if not np.isnan(alt := geometry_reconstructor.alt)
            else event.pointing.array_altitude
        )
        x = x if not np.isnan(x := geometry_reconstructor.core_x) else 0 * u.m
        y = y if not np.isnan(y := geometry_reconstructor.core_y) else 0 * u.m
        width = 20 * u.m
        length = 3000 * u.m
        h_max = (
            h_max
            if not np.isnan(h_max := geometry_reconstructor.h_max)
            else 10000 * u.m
        )

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
