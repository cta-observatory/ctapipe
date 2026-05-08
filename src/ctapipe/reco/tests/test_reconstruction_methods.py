import pytest
from astropy import units as u
from traitlets.config import Config

from ctapipe.calib import CameraCalibrator
from ctapipe.image import ImageProcessor
from ctapipe.io import EventSource
from ctapipe.reco import HillasIntersection, HillasReconstructor
from ctapipe.reco.impact import ImPACTReconstructor
from ctapipe.reco.impact_utilities import create_dummy_templates
from ctapipe.utils import get_dataset_path

reconstructors = [HillasIntersection, HillasReconstructor, ImPACTReconstructor]


@pytest.mark.parametrize("cls", reconstructors)
def test_reconstructors(cls, tmp_path):
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

    source = EventSource(filename, max_events=4, focal_length_choice="EQUIVALENT")
    subarray = source.subarray
    calib = CameraCalibrator(source.subarray)
    image_processor = ImageProcessor(source.subarray)

    for event in source:
        calib(event)
        image_processor(event)

        reco_config = Config()

        if cls is ImPACTReconstructor:
            pytest.importorskip("iminuit")

            for tel_type in subarray.telescope_types:
                create_dummy_templates(
                    str(tmp_path) + "/{}.template.gz".format(str(tel_type)),
                    1,
                    str(tel_type),
                )

            reco_config = Config(
                {
                    "ImPACTReconstructor": {
                        "image_template_path": [
                            [
                                "type",
                                "LST_LST_LSTCam",
                                str(tmp_path) + "/LST_LST_LSTCam.template.gz",
                            ],
                            [
                                "type",
                                "MST_MST_NectarCam",
                                str(tmp_path) + "/MST_MST_NectarCam.template.gz",
                            ],
                        ],
                        "pedestal_width": [
                            ["type", "LST_LST_LSTCam", 1.0],
                            ["type", "MST_MST_NectarCam", 1.0],
                        ],
                        "spe_width": [
                            ["type", "LST_LST_LSTCam", 1.0],
                            ["type", "MST_MST_NectarCam", 1.0],
                        ],
                    }
                }
            )

        reconstructor = cls(
            subarray,
            atmosphere_profile=source.atmosphere_density_profile,
            config=reco_config,
        )
        reconstructor(event)

        name = cls.__name__
        # test the container is actually there and not only created by Map
        assert name in event.dl2.stereo.geometry
        assert event.dl2.stereo.geometry[name].alt.unit.is_equivalent(u.deg)
        assert event.dl2.stereo.geometry[name].az.unit.is_equivalent(u.deg)
        assert event.dl2.stereo.geometry[name].core_x.unit.is_equivalent(u.m)
