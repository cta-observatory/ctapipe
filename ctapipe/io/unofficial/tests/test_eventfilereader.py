import pytest
from ctapipe.io.unofficial import eventfilereader as uefr
from ctapipe.utils import get_dataset


def test_check_modules_installed():
    assert(not uefr.check_modules_installed(["unlikely_module_name"]))
    assert(not uefr.check_modules_installed(["unlikely_module_name", 'numpy']))
    assert(uefr.check_modules_installed(["numpy"]))


@pytest.mark.skipif(not uefr.check_modules_installed(uefr.targetio_modules),
                    reason="Requires targetio specific modules")
def test_targetiofilereader():
    url = get_dataset("chec_r1.tio")
    assert(uefr.TargetioFileReader.check_file_compatibility(url))
    reader = uefr.TargetioFileReader(None, None, input_path=url)
    assert(reader.origin == 'targetio')
    assert(reader.num_events == 5)
    reader.max_events = 3
    assert(reader.num_events == 3)
    source = reader.read()
    event = next(source)
    assert(round(event.r1.tel[0].pe_samples[0, 0, 0]) == -274)
    event = reader.get_event(0)
    assert(round(event.r1.tel[0].pe_samples[0, 0, 0]) == -274)
