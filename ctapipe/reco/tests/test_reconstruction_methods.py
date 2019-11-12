from astropy import units as u
import numpy as np

from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.io import event_source
from ctapipe.reco.HillasReconstructor import HillasReconstructor
from ctapipe.reco.hillas_intersection import HillasIntersection

from ctapipe.utils import get_dataset_path
from astropy.coordinates import SkyCoord, AltAz

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

    in the end, proper units in the output are asserted """

    filename = get_dataset_path("gamma_test_large.simtel.gz")

    source = event_source(filename, max_events=10)
    horizon_frame = AltAz()

    # record how many events were reconstructed by each reconstructor
    reconstructed_events = np.zeros((len(reconstructors)))

    for event in source:
        array_pointing = SkyCoord(
            az=event.mc.az,
            alt=event.mc.alt,
            frame=horizon_frame
        )

        hillas_dict = {}
        telescope_pointings = {}

        for tel_id in event.dl0.tels_with_data:

            geom = event.inst.subarray.tel[tel_id].camera

            telescope_pointings[tel_id] = SkyCoord(alt=event.pointing[tel_id].altitude,
                                                   az=event.pointing[tel_id].azimuth,
                                                   frame=horizon_frame)
            pmt_signal = event.r0.tel[tel_id].waveform[0].sum(axis=1)

            mask = tailcuts_clean(geom, pmt_signal,
                                  picture_thresh=10., boundary_thresh=5.)
            pmt_signal[mask == 0] = 0

            try:
                moments = hillas_parameters(geom, pmt_signal)
                hillas_dict[tel_id] = moments
            except HillasParameterizationError as e:
                print(e)
                continue

        if len(hillas_dict) < 2:
            continue

        for count, reco_method in enumerate(reconstructors):
            reconstructed_events[count] += 1
            reconstructor = reco_method()
            reconstructor_out = reconstructor.predict(hillas_dict, event.inst, array_pointing, telescope_pointings)

            reconstructor_out.alt.to(u.deg)
            reconstructor_out.az.to(u.deg)
            reconstructor_out.core_x.to(u.m)
            assert reconstructor_out.is_valid

    np.testing.assert_array_less(np.zeros_like(reconstructed_events), reconstructed_events)
