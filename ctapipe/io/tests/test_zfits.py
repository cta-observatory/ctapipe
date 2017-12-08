import pytest
from ctapipe.utils.datasets import get_dataset
import numpy as np

@pytest.mark.skip
def test_loop_over_events():
    from ctapipe.io.zfits import zfits_event_source

    _url = get_dataset("sst-1m_5evts.fits.fz")
    inputfile_reader = zfits_event_source(url=_url, max_events= 5 )
    i=0
    for event in inputfile_reader:
        tels = event.r0.tels_with_data
        assert tels == [3]
        for telid in event.r0.tels_with_data:
            evt_num = event.r0.tel[telid].camera_event_number
            assert i == evt_num
            adcs = np.array(list(event.r0.tel[telid].adc_samples.values()))
            assert adcs.shape == (1296,20,)
        i+=1
