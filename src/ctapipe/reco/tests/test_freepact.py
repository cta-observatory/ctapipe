import os

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, Angle
from traitlets.config import Config

from ctapipe.containers import (
    HillasParametersContainer,
    ReconstructedEnergyContainer,
    ReconstructedGeometryContainer,
)
from ctapipe.reco.freepact import FreePACTReconstructor, create_dummy_freepact_templates
from ctapipe.utils import get_dataset_path

pytest.importorskip("iminuit")
pytest.importorskip("tensorflow")

SIMTEL_PATH = get_dataset_path(
    "gamma_20deg_0deg_run2___cta-prod5-paranal_desert"
    "-2147m-Paranal-dark_cone10-100evts.simtel.zst"
)


def get_simtel_profile_from_eventsource():
    """get a TableAtmosphereDensityProfile from a simtel file"""
    from ctapipe.io import EventSource

    with EventSource(SIMTEL_PATH) as source:
        return source.atmosphere_density_profile


@pytest.fixture(scope="session")
def table_profile():
    """a table profile for testing"""
    return get_simtel_profile_from_eventsource()


class TestFreePACT:
    @classmethod
    def setup_class(self):
        self.horizon_frame = AltAz()

        self.h1 = HillasParametersContainer(
            fov_lon=1 * u.deg,
            fov_lat=1 * u.deg,
            r=1 * u.deg,
            phi=Angle(0 * u.deg),
            intensity=100,
            length=0.4 * u.deg,
            width=0.4 * u.deg,
            psi=Angle(0 * u.deg),
            skewness=0,
            kurtosis=0,
        )

    def test_interpolation(self, tmp_path, example_subarray, table_profile):
        """Test interpolation works on dummy template library"""

        for tel_type in example_subarray.telescope_types:
            os.mkdir(str(tmp_path) + "/" + str(tel_type)[0:3] + "/")
            create_dummy_freepact_templates(
                str(tmp_path) + "/" + str(tel_type)[0:3] + "/",
                str(tel_type)[0:3],
                0,
                0,
                0,
            )
            create_dummy_freepact_templates(
                str(tmp_path) + "/" + str(tel_type)[0:3] + "/",
                str(tel_type)[0:3],
                20,
                0,
                0,
            )
            create_dummy_freepact_templates(
                str(tmp_path) + "/" + str(tel_type)[0:3] + "/",
                str(tel_type)[0:3],
                0,
                180,
                0,
            )
            create_dummy_freepact_templates(
                str(tmp_path) + "/" + str(tel_type)[0:3] + "/",
                str(tel_type)[0:3],
                20,
                180,
                0,
            )

            freepact_config = Config(
                {
                    "FreePACTReconstructor": {
                        "image_template_path": [
                            [
                                "type",
                                "*",
                                str(tmp_path) + "/SST/",
                            ],
                        ]
                    }
                }
            )

            freepact = FreePACTReconstructor(
                example_subarray, table_profile, config=freepact_config
            )

            pred = freepact.prediction[list(freepact.prediction.keys())[0]](
                0,
                0,
                np.array([1, 1]),
                np.array([100, 100]),
                np.array([300, 300]),
                np.array([[0, 0], [0, 0]]),
                np.array([[0, 1], [0, 0]]),
                np.array([[1, 1], [0, 0]]),
            )
            assert pred.shape == (2, 2)


def test_selected_subarray(
    subarray_and_event_gamma_off_axis_500_gev, tmp_path, table_profile
):
    subarray, event = subarray_and_event_gamma_off_axis_500_gev

    for tel_type in subarray.telescope_types:
        os.mkdir(str(tmp_path) + "/" + str(tel_type)[0:3] + "/")
        create_dummy_freepact_templates(
            str(tmp_path) + "/" + str(tel_type)[0:3] + "/", str(tel_type)[0:3], 0, 0, 0
        )
        create_dummy_freepact_templates(
            str(tmp_path) + "/" + str(tel_type)[0:3] + "/", str(tel_type)[0:3], 60, 0, 0
        )
        create_dummy_freepact_templates(
            str(tmp_path) + "/" + str(tel_type)[0:3] + "/",
            str(tel_type)[0:3],
            0,
            180,
            0,
        )
        create_dummy_freepact_templates(
            str(tmp_path) + "/" + str(tel_type)[0:3] + "/",
            str(tel_type)[0:3],
            60,
            180,
            0,
        )
        print(str(tmp_path) + "/" + str(tel_type)[0:3] + "/")
    freepact_config = Config(
        {
            "FreePACTReconstructor": {
                "image_template_path": [
                    [
                        "type",
                        "LST_LST_LSTCam",
                        str(tmp_path) + "/LST/",
                    ],
                ],
                "pedestal_width": [["type", "LST_LST_LSTCam", 1.0]],
                "spe_width": [["type", "LST_LST_LSTCam", 1.0]],
            }
        }
    )

    shower_test = ReconstructedGeometryContainer()
    energy_test = ReconstructedEnergyContainer()
    shower_test.prefix = "test"
    energy_test.prefix = "test_energy"
    # Transform everything back to a useful system
    shower_test.alt, shower_test.az = 70 * u.deg, 0 * u.deg

    shower_test.core_x = -10 * u.m
    shower_test.core_y = -10 * u.m
    shower_test.core_tilted_x = -10 * u.m
    shower_test.core_tilted_y = -10 * u.m

    shower_test.is_valid = True

    energy_test.energy = 0.5 * u.TeV
    energy_test.is_valid = True

    event.dl2.stereo.geometry["test"] = shower_test
    event.dl2.stereo.energy["test_energy"] = energy_test

    freepact_reco = FreePACTReconstructor(
        subarray, table_profile, config=freepact_config
    )
    freepact_reco(event)  # effective focal length returning NaN
    print(event.dl2.stereo.geometry["FreePACTReconstructor"])

    assert event.dl2.stereo.geometry["FreePACTReconstructor"].is_valid
    assert event.dl2.stereo.energy["FreePACTReconstructor"].is_valid
