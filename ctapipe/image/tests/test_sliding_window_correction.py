import numpy as np
import astropy.units as u
from numpy.testing import assert_allclose

from ctapipe.image.extractor import SlidingWindowMaxSum
from ctapipe.image.toymodel import WaveformModel
from ctapipe.instrument import SubarrayDescription, TelescopeDescription

from ctapipe.utils import datasets


def test_sw_pulse_lst(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(
            datasets,
            "DEFAULT_URL",
            "http://cccta-dataserver.in2p3.fr/data/ctapipe-extra/v0.3.2/",
        )

        # prepare array with 1 LST
        subarray = SubarrayDescription(
            "LST1",
            tel_positions={1: np.zeros(3) * u.m},
            tel_descriptions={
                1: TelescopeDescription.from_name(
                    optics_name="LST", camera_name="LSTCam"
                )
            },
        )

        telid = list(subarray.tel.keys())[0]

        n_pixels = subarray.tel[telid].camera.geometry.n_pixels
        n_samples = 40
        readout = subarray.tel[telid].camera.readout

        random = np.random.RandomState(1)
        minCharge = 100
        maxCharge = 1000
        charge_true = random.uniform(minCharge, maxCharge, n_pixels)
        time_true = random.uniform(
            n_samples // 2 - 1, n_samples // 2 + 1, n_pixels
        ) / readout.sampling_rate.to_value(u.GHz)

        waveform_model = WaveformModel.from_camera_readout(readout)
        waveform = waveform_model.get_waveform(charge_true, time_true, n_samples)
        selected_gain_channel = np.zeros(charge_true.size, dtype=np.int)

        # define extractor
        extr_width = 8
        extractor = SlidingWindowMaxSum(subarray=subarray)
        extractor.window_width = extr_width
        charge, peak_time = extractor(waveform, telid, selected_gain_channel)
        print(charge / charge_true)
        assert_allclose(charge, charge_true, rtol=0.02)
