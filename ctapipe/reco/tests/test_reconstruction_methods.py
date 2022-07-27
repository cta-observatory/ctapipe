import pytest
from astropy import units as u

from ctapipe.calib import CameraCalibrator
from ctapipe.image import ImageProcessor
from ctapipe.io import EventSource

from ctapipe.reco.HillasReconstructor import HillasReconstructor
from ctapipe.reco.hillas_intersection import HillasIntersection
from ctapipe.reco.ImPACT import ImPACTReconstructor
from ctapipe.containers import ReconstructedEnergyContainer

from ctapipe.utils import get_dataset_path


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

    source = EventSource(filename, max_events=10, focal_length_choice="EQUIVALENT")
    subarray = source.subarray
    calib = CameraCalibrator(source.subarray)
    image_processor = ImageProcessor(source.subarray)

    for event in source:
        calib(event)
        image_processor(event)

        for ReconstructorType in reconstructors:
            reconstructor = ReconstructorType(subarray)

            reconstructor(event)
            
            name = ReconstructorType.__name__
            assert event.dl2.stereo.geometry[name].alt.unit.is_equivalent(u.deg)
            assert event.dl2.stereo.geometry[name].az.unit.is_equivalent(u.deg)
            assert event.dl2.stereo.geometry[name].core_x.unit.is_equivalent(u.m)

                

def test_ImPACT(reconstructors):
    """
    a test of the complete fit procedure on one event including:
    • tailcut cleaning
    • hillas parametrisation
    • HillasPlane creation
    • direction fit
    • position fit

    in the end, proper units in the output are asserted"""

    filename = get_dataset_path("gamma_test_large.simtel.gz")

    source = EventSource(filename, max_events=10)
    calib = CameraCalibrator(source.subarray)
    horizon_frame = AltAz()

    # record how many events were reconstructed by each reconstructor
    reconstructed_events = np.zeros((len(reconstructors)))
    impact_reconstructor = ImPACTReconstructor(dummy_reconstructor=True)

    for event in source:
        calib(event)
        sim_shower = event.simulation.shower
        array_pointing = SkyCoord(
            az=sim_shower.az, alt=sim_shower.alt, frame=horizon_frame
        )

        hillas_dict, image_dict, mask_dict = {}, {}, {}
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

            try:
                moments = hillas_parameters(geom[mask], dl1.image[mask])
                hillas_dict[tel_id] = moments
                mask_dict[tel_id] = mask
                image_dict[tel_id] = dl1.image
            except HillasParameterizationError as e:
                print(e)
                continue

        if len(hillas_dict) < 2:
            continue

        for count, reco_method in enumerate(reconstructors):

            reconstructed_events[count] += 1
            reconstructor = reco_method()
            reconstructor_out = reconstructor.predict(
                hillas_dict, source.subarray, array_pointing, telescope_pointings
            )

            energy_seed = ReconstructedEnergyContainer()
            energy_seed.is_valid = True
            energy_seed.energy = 1 * u.TeV

            shower_result, energy_result = impact_reconstructor.predict(
                hillas_dict, source.subarray, array_pointing, telescope_pointings,
                image_dict, mask_dict, reconstructor_out, energy_seed
            )

            shower_result.alt.to(u.deg)
            shower_result.az.to(u.deg)
            shower_result.core_x.to(u.m)
            assert shower_result.is_valid

    np.testing.assert_array_less(
        np.zeros_like(reconstructed_events), reconstructed_events
    )
