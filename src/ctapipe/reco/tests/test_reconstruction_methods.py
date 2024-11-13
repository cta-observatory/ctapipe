import pytest
from astropy import units as u

from ctapipe.calib import CameraCalibrator
from ctapipe.image import ImageProcessor
from ctapipe.io import EventSource
from ctapipe.reco import HillasIntersection, HillasReconstructor
from ctapipe.reco.impact import ImPACTReconstructor
from ctapipe.utils import get_dataset_path

reconstructors = [HillasIntersection, HillasReconstructor, ImPACTReconstructor]


@pytest.mark.parametrize("cls", reconstructors)
def test_reconstructors(cls):
    """
    a test of the complete fit procedure on one event including:
    • tailcut cleaning
    • hillas parametrisation
    • direction fit
    • position fit
    in the end, proper units in the output are asserted"""

    filename = get_dataset_path(
        "gamma_LaPalma_baseline_20Zd_180Az_prod3b_test.simtel.gz"
    )
    template_file = get_dataset_path("LSTCam.template.gz")

    source = EventSource(filename, max_events=4, focal_length_choice="EQUIVALENT")
    subarray = source.subarray
    calib = CameraCalibrator(source.subarray)
    image_processor = ImageProcessor(source.subarray)

    for event in source:
        calib(event)
        image_processor(event)

        if cls is ImPACTReconstructor:
            pytest.importorskip("iminuit")

        reconstructor = cls(
            subarray, atmosphere_profile=source.atmosphere_density_profile
        )
        if cls is ImPACTReconstructor:
            reconstructor.root_dir = str(template_file.parents[0])

        reconstructor(event)

        name = cls.__name__
        # test the container is actually there and not only created by Map
        assert name in event.dl2.stereo.geometry
        assert event.dl2.stereo.geometry[name].alt.unit.is_equivalent(u.deg)
        assert event.dl2.stereo.geometry[name].az.unit.is_equivalent(u.deg)
        assert event.dl2.stereo.geometry[name].core_x.unit.is_equivalent(u.m)
