import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ctapipe.containers import (
    ReconstructedShowerContainer,
    ReconstructedEnergyContainer,
)
from ctapipe.reco.ImPACT import ImPACTReconstructor
from ctapipe.reco.ImPACT_utilities import *
from ctapipe.containers import HillasParametersContainer
from astropy.coordinates import Angle, AltAz, SkyCoord

from ctapipe.utils import get_dataset_path
from ctapipe.io import EventSource


class TestImPACT:
    @classmethod
    def setup_class(self):
        self.impact_reco = ImPACTReconstructor(root_dir=".", dummy_reconstructor=True)
        self.horizon_frame = AltAz()

        self.h1 = HillasParametersContainer(
            x=1 * u.deg,
            y=1 * u.deg,
            r=1 * u.deg,
            phi=Angle(0 * u.deg),
            intensity=100,
            length=0.4 * u.deg,
            width=0.4 * u.deg,
            psi=Angle(0 * u.deg),
            skewness=0,
            kurtosis=0,
        )

        #filename = get_dataset_path("gamma_test_large.simtel.gz")
        #source = EventSource(filename, max_events=1)

        #self.subarray = source.subarray

    def test_brightest_mean_average(self):
        """
        Test that averaging of the brightest pixel position give a sensible outcome
        """
        image = np.array([1, 1, 1, 1])
        pixel_x = np.array([0.0, 1.0, 0.0, -1.0]) * u.deg
        pixel_y = np.array([-1.0, 0.0, 1.0, 0.0]) * u.deg

        self.impact_reco.hillas_parameters = [self.h1]
        self.impact_reco.pixel_x = [pixel_x]
        self.impact_reco.pixel_y = [pixel_y]

        self.impact_reco.get_hillas_mean()

        assert_allclose(
            self.impact_reco.peak_x[0] * (180 / np.pi), 1, rtol=0, atol=0.001
        )
        assert_allclose(
            self.impact_reco.peak_y[0] * (180 / np.pi), 1, rtol=0, atol=0.001
        )

    def test_rotation(self):
        """Test pixel rotation function"""
        x = np.array([1])
        y = np.array([0])

        xt, yt = rotate_translate(x, y, 0, 0, np.deg2rad(90))
        assert_allclose(xt, 0, rtol=0, atol=0.001)
        assert_allclose(yt, 1, rtol=0, atol=0.001)

        xt, yt = rotate_translate(x, y, 0, 0, np.deg2rad(180))
        assert_allclose(xt, 1, rtol=0, atol=0.001)
        assert_allclose(yt, 0, rtol=0, atol=0.001)

    def test_translation(self):
        """Test pixel translation function"""
        x = np.array([0])
        y = np.array([0])

        xt, yt = rotate_translate(x, y, 1, 1, np.array([0]))
        assert_allclose(xt, 1, rtol=0, atol=0.001)
        assert_allclose(yt, -1, rtol=0, atol=0.001)

    def test_xmax_calculation(self):
        """Test calculation of hmax and interpolation of Xmax tables"""

        image = np.array([1, 1, 1])
        pixel_x = np.array([1, 1, 1]) * u.deg
        pixel_y = np.array([1, 1, 1]) * u.deg

        self.impact_reco.hillas_parameters = [self.h1]
        self.impact_reco.pixel_x = np.array([pixel_x])
        self.impact_reco.pixel_y = np.array([pixel_y])
        self.impact_reco.tel_pos_x = np.array([0.])
        self.impact_reco.tel_pos_y = np.array([0.])

        self.impact_reco.get_hillas_mean()

        shower_max = self.impact_reco.get_shower_max(0, 0, 0, 100, 0)
        assert_allclose(shower_max, 484.2442217190515, rtol=0.01)

    @pytest.mark.skip("need a dataset for this to work")
    def test_likelihood(self):

        image = np.array([1, 1, 1])
        pixel_x = np.array([1, 1, 1]) * u.deg
        pixel_y = np.array([1, 1, 1]) * u.deg
        array_pointing = SkyCoord(alt=0 * u.deg, az=0 * u.deg, 
                                  frame=self.horizon_frame)
        
        self.impact_reco.tel_types = np.array(["LSTCam"])
        self.impact_reco.initialise_templates({1: "LSTCam"})

        self.impact_reco.image = image
        self.impact_reco.hillas_parameters = [self.h1]
        self.impact_reco.pixel_x = pixel_x
        self.impact_reco.pixel_y = pixel_y
        self.impact_reco.tel_pos_x = np.array([0.])
        self.impact_reco.tel_pos_y = np.array([0.])
        self.impact_reco.array_direction = array_pointing

        self.impact_reco.get_hillas_mean()
        like = self.impact_reco.get_likelihood(0, 0, 0, 100, 1, 0)
        assert like is not np.nan and like > 0
