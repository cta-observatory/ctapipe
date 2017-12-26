import pytest
from ctapipe.io.unofficial import eventfilereader as uefr
from ctapipe.utils import get_dataset


def test_targetiofilereader():
    pytest.importorskip("target_driver")
    pytest.importorskip("target_io")
    pytest.importorskip("target_calib")
    url = get_dataset("chec_r1.tio")
    assert(uefr.TargetioFileReader.check_file_compatibility(url))
    reader = uefr.TargetioFileReader(None, None, input_path=url)
    assert(reader.num_events == 5)
    reader.max_events = 3
    assert(reader.num_events == 3)
    source = reader.read()
    event = next(source)
    assert(round(event.r1.tel[0].pe_samples[0, 0, 0]) == -274)
    event = reader.get_event(0)
    assert(round(event.r1.tel[0].pe_samples[0, 0, 0]) == -274)
