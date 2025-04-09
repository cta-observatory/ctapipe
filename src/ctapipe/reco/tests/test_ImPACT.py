import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, Angle, SkyCoord
from numpy.testing import assert_allclose

from ctapipe.containers import (
    HillasParametersContainer,
    ReconstructedEnergyContainer,
    ReconstructedGeometryContainer,
)
from ctapipe.reco.impact import ImPACTReconstructor
from ctapipe.reco.impact_utilities import (
    create_dummy_templates,
    create_seed,
    generate_fake_template,
    rotate_translate,
)
from ctapipe.utils import get_dataset_path

pytest.importorskip("iminuit")

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


class TestImPACT:
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

    def test_brightest_mean_average(self, example_subarray, table_profile):
        """
        Test that averaging of the brightest pixel position give a sensible outcome
        """

        impact_reco = ImPACTReconstructor(example_subarray, table_profile)
        pixel_x = np.array([0.0, 1.0, 0.0, -1.0]) * u.deg
        pixel_y = np.array([-1.0, 0.0, 1.0, 0.0]) * u.deg

        impact_reco.hillas_parameters = [self.h1]
        impact_reco.pixel_x = [pixel_x]
        impact_reco.pixel_y = [pixel_y]

        impact_reco.get_hillas_mean()

        assert_allclose(impact_reco.peak_x[0] * (180 / np.pi), 1, rtol=0, atol=0.001)
        assert_allclose(impact_reco.peak_y[0] * (180 / np.pi), 1, rtol=0, atol=0.001)

    def test_rotation(self):
        """Test pixel rotation function"""
        x = np.array([[1]])
        y = np.array([[0]])

        xt, yt = rotate_translate(x, y, 0, 0, np.deg2rad(np.array([90])))
        assert_allclose(xt, 0, rtol=0, atol=0.001)
        assert_allclose(yt, 1, rtol=0, atol=0.001)

        xt, yt = rotate_translate(x, y, 0, 0, np.deg2rad(np.array([180])))
        assert_allclose(xt, 1, rtol=0, atol=0.001)
        assert_allclose(yt, 0, rtol=0, atol=0.001)

    def test_translation(self):
        """Test pixel translation function"""
        x = np.array([[0]])
        y = np.array([[0]])

        xt, yt = rotate_translate(x, y, 1, 1, np.array([0]))
        assert_allclose(xt, 1, rtol=0, atol=0.001)
        assert_allclose(yt, -1, rtol=0, atol=0.001)

    def test_xmax_calculation(self, example_subarray, table_profile):
        """Test calculation of hmax and interpolation of Xmax tables"""

        impact_reco = ImPACTReconstructor(example_subarray, table_profile)
        pixel_x = np.array([1, 1, 1]) * u.deg
        pixel_y = np.array([1, 1, 1]) * u.deg

        impact_reco.hillas_parameters = [self.h1]
        impact_reco.pixel_x = np.array([pixel_x])
        impact_reco.pixel_y = np.array([pixel_y])
        impact_reco.tel_pos_x = np.array([0.0])
        impact_reco.tel_pos_y = np.array([0.0])

        impact_reco.get_hillas_mean()

        shower_max = impact_reco.get_shower_max(0, 0, 0, 100, 0)
        assert_allclose(shower_max, 489.525278, rtol=0.01)

    def test_interpolation(self, tmp_path, example_subarray, table_profile):
        """Test interpolation works on dummy template library"""

        impact_reco = ImPACTReconstructor(example_subarray, table_profile)

        create_dummy_templates(str(tmp_path) + "/dummy.template.gz", 1)
        template, x, y = generate_fake_template(-1.5, 0.5)
        template *= 1000

        impact_reco.root_dir = str(tmp_path)
        impact_reco.initialise_templates({1: "dummy"})

        pred = impact_reco.image_prediction(
            "dummy",
            0,
            0,
            np.array([1]),
            np.array([100]),
            np.array([-150]),
            x.ravel(),
            y.ravel(),
        )

        assert_allclose(template.ravel() - pred, np.zeros_like(pred), atol=0.1)

    def test_fitting(self, tmp_path, example_subarray, table_profile):
        impact_reco = ImPACTReconstructor(example_subarray, table_profile)

        create_dummy_templates(str(tmp_path) + "/dummy.template.gz", 1)
        impact_reco.root_dir = str(tmp_path)

        tel1, x, y = generate_fake_template(-1.5, 0.5, 0.3, 50, 50, ((-4, 4), (-4, 4)))
        tel2 = np.rot90(tel1)
        tel3 = np.rot90(tel2)
        tel4 = np.rot90(tel3)

        image = np.array([tel1.ravel(), tel2.ravel(), tel3.ravel(), tel4.ravel()])
        pixel_x = np.array([x.ravel(), x.ravel(), x.ravel(), x.ravel()]) * u.deg
        pixel_y = np.array([y.ravel(), y.ravel(), y.ravel(), y.ravel()]) * u.deg

        array_pointing = SkyCoord(alt=0 * u.deg, az=0 * u.deg, frame=AltAz)

        impact_reco.tel_types = np.array(["dummy", "dummy", "dummy", "dummy"])
        impact_reco.initialise_templates(
            {1: "dummy", 2: "dummy", 3: "dummy", 4: "dummy"}
        )
        impact_reco.zenith = 0  # *u.deg
        impact_reco.azimuth = 0  # *u.deg
        impact_reco.ped = np.ones_like(image)  # *u.deg

        impact_reco.image = image * 1000
        impact_reco.hillas_parameters = [self.h1, self.h1, self.h1, self.h1]
        impact_reco.pixel_x = np.ma.array(np.deg2rad(pixel_x))
        impact_reco.pixel_y = np.ma.array(np.deg2rad(pixel_y))
        impact_reco.tel_pos_x = np.array([0, 100, -0, -100])
        impact_reco.tel_pos_y = np.array([-100.0, 0, 100, 0])
        impact_reco.array_direction = array_pointing

        impact_reco.get_hillas_mean()

        seed, step, limits = create_seed(0.0, 0.0, 0.0, 0.0, 0.5)
        vals, error, chi2 = impact_reco.minimise(seed, step, limits)
        assert_allclose(vals[4], 1, rtol=0.05)

        vals, error, chi2 = impact_reco.minimise(seed, step, limits)
        assert_allclose(vals[4], 1, rtol=0.05)
        theta = np.sqrt(vals[0] ** 2 + vals[1] ** 2)
        assert_allclose(np.rad2deg(theta), 0, atol=0.02)


def test_selected_subarray(
    subarray_and_event_gamma_off_axis_500_gev, tmp_path, table_profile
):
    """test that reconstructor also works with "missing" ids"""

    create_dummy_templates(str(tmp_path) + "/LSTCam.template.gz", 1)

    subarray, event = subarray_and_event_gamma_off_axis_500_gev

    shower_test = ReconstructedGeometryContainer()
    energy_test = ReconstructedEnergyContainer()
    shower_test.prefix = "test"
    energy_test.prefix = "test_energy"
    # Transform everything back to a useful system
    shower_test.alt, shower_test.az = 70 * u.deg, 0 * u.deg

    shower_test.core_x = 0 * u.m
    shower_test.core_y = 0 * u.m
    shower_test.core_tilted_x = 0 * u.m
    shower_test.core_tilted_y = 0 * u.m

    shower_test.is_valid = True

    energy_test.energy = 500 * u.GeV
    energy_test.is_valid = True

    event.dl2.stereo.geometry["test"] = shower_test
    event.dl2.stereo.energy["test_energy"] = energy_test
    reconstructor = ImPACTReconstructor(subarray, table_profile)
    reconstructor.root_dir = str(tmp_path)
    reconstructor(event)
    assert event.dl2.stereo.geometry["ImPACTReconstructor"].is_valid
    assert event.dl2.stereo.energy["ImPACTReconstructor"].is_valid
