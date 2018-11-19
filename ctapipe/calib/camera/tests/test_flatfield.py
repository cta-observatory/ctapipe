import numpy as np
import pytest

from ctapipe.calib.camera.flatfield import FlasherFlatFieldCalculator




def test_FlasherFlatFieldCalculator():
    ff_calculator = FlasherFlatFieldCalculator()

    for chan in [0, 1]:
        gs.channel = chan

        waveforms_2g = np.random.normal(size=(2, 1000, 30))
        waveforms_1g, mask = gs.select_gains("NectarCam", waveforms_2g)

        assert waveforms_1g.shape == (1000, 30)
        assert (waveforms_1g == waveforms_2g[chan]).all()
        assert mask.shape == (1000,)
