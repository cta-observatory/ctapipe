from astropy import units as u
import numpy as np

from ctapipe.containers import ImageParametersContainer, HillasParametersContainer
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.io import EventSource
from ctapipe.reco import HillasReconstructor, HillasIntersection

from ctapipe.reco.reco_algorithms import InvalidWidthException

from ctapipe.utils import get_dataset_path
from astropy.coordinates import SkyCoord, AltAz
from ctapipe.calib import CameraCalibrator

import pytest


@pytest.fixture
def reconstructors():
    return [HillasIntersection, HillasReconstructor]


def test_reconstructors(reconstructors):
    """
    a test of the complete fit procedure on one event including:
    • tailcut cleaning
    • hillas parametrisation
    • HillasPlane creation
    • direction fit
    • position fit

    in the end, proper units in the output are asserted"""

    filename = get_dataset_path(
        "gamma_LaPalma_baseline_20Zd_180Az_prod3b_test.simtel.gz"
    )

    source = EventSource(filename, max_events=10)
    subarray = source.subarray
    calib = CameraCalibrator(source.subarray)
    horizon_frame = AltAz()

    for event in source:
        calib(event)
        sim_shower = event.simulation.shower
        array_pointing = SkyCoord(
            az=sim_shower.az, alt=sim_shower.alt, frame=horizon_frame
        )

        telescope_pointings = {}

        for tel_id, dl1 in event.dl1.tel.items():

            geom = source.subarray.tel[tel_id].camera.geometry

            telescope_pointings[tel_id] = SkyCoord(
                alt=event.pointing.tel[tel_id].altitude,
                az=event.pointing.tel[tel_id].azimuth,
                frame=horizon_frame,
            )
            mask = tailcuts_clean(
                geom, dl1.image, picture_thresh=10.0, boundary_thresh=5.0
            )

            dl1.parameters = ImageParametersContainer()

            try:
                moments = hillas_parameters(geom[mask], dl1.image[mask])
            except HillasParameterizationError:
                dl1.parameters.hillas = HillasParametersContainer()
                continue

            # Make sure we provide only good images for the test
            if np.isnan(moments.width.value) or (moments.width.value == 0):
                dl1.parameters.hillas = HillasParametersContainer()
            else:
                dl1.parameters.hillas = moments

        hillas_dict = {
            tel_id: dl1.parameters.hillas
            for tel_id, dl1 in event.dl1.tel.items()
            if np.isfinite(dl1.parameters.hillas.intensity)
        }
        if len(hillas_dict) < 2:
            continue

        for count, reco_method in enumerate(reconstructors):
            if reco_method is HillasReconstructor:
                reconstructor = HillasReconstructor(subarray)

                reconstructor(event)

                event.dl2.stereo.geometry["HillasReconstructor"].alt.to(u.deg)
                event.dl2.stereo.geometry["HillasReconstructor"].az.to(u.deg)
                event.dl2.stereo.geometry["HillasReconstructor"].core_x.to(u.m)
                assert event.dl2.stereo.geometry["HillasReconstructor"].is_valid

            else:
                reconstructor = reco_method()

                try:
                    reconstructor_out = reconstructor.predict(
                        hillas_dict,
                        source.subarray,
                        array_pointing,
                        telescope_pointings,
                    )
                except InvalidWidthException:
                    continue

                reconstructor_out.alt.to(u.deg)
                reconstructor_out.az.to(u.deg)
                reconstructor_out.core_x.to(u.m)
                assert reconstructor_out.is_valid
