from ctapipe.utils.datasets import get_datasets_path
from ctapipe.io.zfits import zfits_event_source
import numpy as np

def test_loop_over_events():
    _url = get_datasets_path("sst-1m_5evts.fits.fz")
    inputfile_reader = zfits_event_source(url=_url, max_events= 5 )
    i=0
    for event in inputfile_reader:
        tels = event.dl0.tels_with_data
        assert tels == [3]
        for telid in event.dl0.tels_with_data:
            evt_num = event.dl0.tel[telid].event_number
            assert i == evt_num
            adcs = np.array(list(event.dl0.tel[telid].adc_samples.values()))
            assert adcs.shape == (1296,20,)
        i+=1
