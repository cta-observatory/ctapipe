import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ctapipe.io.containers import (ReconstructedShowerContainer,
                                   ReconstructedEnergyContainer)
from ctapipe.reco.ImPACT import ImPACTReconstructor
from ctapipe.io.containers import HillasParametersContainer
from astropy.coordinates import Angle, AltAz, SkyCoord


class TestImPACT():

    @classmethod
    def setup_class(self):
        self.impact_reco = ImPACTReconstructor(root_dir=".")
        self.horizon_frame = AltAz()

        self.h1 = HillasParametersContainer(x=1 * u.deg, y=1 * u.deg,
                                            r=1 * u.deg, phi=Angle(0 * u.rad),
                                            intensity=100,
                                            length=0.4 * u.deg,
                                            width=0.4 * u.deg,
                                            psi=Angle(0 * u.rad),
                                            skewness=0,
                                            kurtosis=0)

    #@pytest.mark.skip('need a dataset for this to work')
    def test_brightest_mean_average(self):
        """
        Test that averaging of the brightest pixel position give a sensible outcome
        """
        image = np.array([1, 1, 1, 1])
        pixel_x = np.array([0., 1., 0., -1.]) * u.deg
        pixel_y = np.array([-1., 0., 1., 0.]) * u.deg

        array_pointing = SkyCoord(alt=0 * u.deg,
                                  az=0 * u.deg,
                                  frame=self.horizon_frame)

        self.impact_reco.set_event_properties({1: image}, {1: image},
                                              {1: pixel_x}, {1: pixel_y},
                                              {1: "DUMMY"}, {1: 0 * u.m},
                                              {1: 0 * u.m},
                                              array_direction=array_pointing,
                                              hillas={1: self.h1})

        self.impact_reco.get_hillas_mean()

        assert_allclose(self.impact_reco.peak_x[0]*(180/np.pi), 1, rtol=0, atol=0.001)
        assert_allclose(self.impact_reco.peak_y[0]*(180/np.pi), 1, rtol=0, atol=0.001)

    def test_rotation(self):
        """Test pixel rotation function"""
        x = np.array([1])
        y = np.array([0])

        xt, yt = ImPACTReconstructor.rotate_translate(x, y, 0, 0,
                                                      np.deg2rad(90))
        assert_allclose(xt, 0, rtol=0, atol=0.001)
        assert_allclose(yt, 1, rtol=0, atol=0.001)

        xt, yt = ImPACTReconstructor.rotate_translate(x, y, 0, 0,
                                                      np.deg2rad(180))
        assert_allclose(xt, 1, rtol=0, atol=0.001)
        assert_allclose(yt, 0, rtol=0, atol=0.001)

    def test_translation(self):
        """Test pixel translation function"""
        x = np.array([0])
        y = np.array([0])

        xt, yt = ImPACTReconstructor.rotate_translate(x, y, 1, 1, np.array([0]))
        assert_allclose(xt, 1, rtol=0, atol=0.001)
        assert_allclose(yt, -1, rtol=0, atol=0.001)

    def test_xmax_calculation(self):
        """Test calculation of hmax and interpolation of Xmax tables"""

        image = np.array([1, 1, 1])
        pixel_x = np.array([1, 1, 1]) * u.deg
        pixel_y = np.array([1, 1, 1]) * u.deg

        array_pointing = SkyCoord(alt=0 * u.deg,
                                  az=0 * u.deg,
                                  frame=self.horizon_frame)

        self.impact_reco.set_event_properties({1: image}, {1: image},
                                              {1: pixel_x},{1: pixel_y},
                                              {1: "DUMMY"}, {1: 0 * u.m},
                                              {1: 0 * u.m},
                                              array_direction=array_pointing,
                                              hillas={1:self.h1})

        shower_max = self.impact_reco.get_shower_max(0, 0, 0, 100, 0)
        assert_allclose(shower_max, 484.2442217190515 , rtol=0.01)

    @pytest.mark.skip('need a dataset for this to work')
    def test_image_prediction(self):
        pixel_x = np.array([0]) * u.deg
        pixel_y = np.array([0]) * u.deg

        image = np.array([1])
        pixel_area = np.array([1]) * u.deg * u.deg

        self.impact_reco.set_event_properties({1: image}, {1: pixel_x},
                                              {1: pixel_y}, {1: pixel_area},
                                              {1: "CHEC"}, {1: 0 * u.m},
                                              {1: 0 * u.m},
                                              array_direction=[0 * u.deg,
                                                               0 * u.deg])

        """First check image prediction by directly accessing the function"""
        pred = self.impact_reco.image_prediction("CHEC", zenith=0, azimuth=0,
                                                 energy=1, impact=50, x_max=0,
                                                 pix_x=pixel_x, pix_y=pixel_y)

        assert np.sum(pred) != 0

        """Then check helper function gives the same answer"""
        shower = ReconstructedShowerContainer()
        shower.is_valid = True
        shower.alt = 0 * u.deg
        shower.az = 0 * u.deg
        shower.core_x = 0 * u.m
        shower.core_y = 100 * u.m
        shower.h_max = 300 + 93 * np.log10(1)

        energy = ReconstructedEnergyContainer()
        energy.is_valid = True
        energy.energy = 1 * u.TeV
        pred2 = self.impact_reco.get_prediction(1, shower_reco=shower,
                                                energy_reco=energy)
        print(pred, pred2)
        assert pred.all() == pred2.all()

    @pytest.mark.skip('need a dataset for this to work')
    def test_likelihood(self):
        pixel_x = np.array([0]) * u.deg
        pixel_y = np.array([0]) * u.deg

        image = np.array([1])
        pixel_area = np.array([1]) * u.deg * u.deg

        self.impact_reco.set_event_properties({1: image}, {1: pixel_x},
                                              {1: pixel_y}, {1: pixel_area},
                                              {1: "CHEC"}, {1: 0 * u.m},
                                              {1: 0 * u.m},
                                              array_direction=[0 * u.deg,
                                                               0 * u.deg])

        like = self.impact_reco.get_likelihood(0, 0, 0, 100, 1, 0)
        assert like is not np.nan and like > 0
