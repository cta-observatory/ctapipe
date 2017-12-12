import pytest
pytest.importorskip("target_driver")
pytest.importorskip("target_io")
pytest.importorskip("target_calib")

from ctapipe.io.unofficial.targetio import targetio
from ctapipe.utils import get_dataset


def test_get_bp_r_c():
    bp, r, c = targetio.get_bp_r_c(200)
    assert(bp == 8)
    assert(r == 6)
    assert(c == 0)


def test_chec_r1():
    url = get_dataset("chec_r1.tio")
    ext = targetio.TargetioExtractor(url)
    ext.read_event(0)
    event = ext.data
    assert(ext.n_pix == 2048)
    assert(ext.r0_samples is None)
    assert(round(ext.r1_samples[0, 0, 0]) == -274)
    assert(event.r0.tels_with_data == {0})
    assert(event.r0.tel[0].adc_samples is None)
    assert(event.r1.tel[0].pe_samples[0, 0, 0] == ext.r1_samples[0, 0, 0])


def test_event_id():
    url = get_dataset("chec_r1.tio")
    ext = targetio.TargetioExtractor(url)
    event_id = 2
    ext.read_event(event_id, True)
    event = ext.data
    assert(event.count == ext.cameraconfig.skip_event - event_id)
    assert(round(ext.r1_samples[0, 0, 0]) == -274)


def test_singlemodule_r0():
    url = get_dataset("targetmodule_r0.tio")
    ext = targetio.TargetioExtractor(url)
    ext.read_event(0)
    event = ext.data
    assert(ext.n_pix == 64)
    assert(round(ext.r0_samples[0, 0, 0]) == 600)
    assert(round(ext.r1_samples[0, 0, 0]) == 0)
    assert(ext.r0_samples.shape[1] == 64)
    assert(event.r0.tels_with_data == {0})
    assert(event.r0.tel[0].adc_samples[0, 0, 0] == ext.r0_samples[0, 0, 0])


def test_singlemodule_r1():
    url = get_dataset("targetmodule_r1.tio")
    ext = targetio.TargetioExtractor(url)
    ext.read_event(0)
    event = ext.data
    assert(ext.n_pix == 64)
    assert(ext.r0_samples is None)
    assert(round(ext.r1_samples[0, 0, 0]) == 0)
    assert(event.r0.tels_with_data == {0})
    assert(event.r0.tel[0].adc_samples is None)
    assert(event.r1.tel[0].pe_samples[0, 0, 0] == ext.r1_samples[0, 0, 0])


def test_singlemodule_r1_source():
    url = get_dataset("targetmodule_r1.tio")
    ext = targetio.TargetioExtractor(url)
    source = ext.read_generator()
    event = next(source)
    assert(ext.n_pix == 64)
    assert(ext.r0_samples is None)
    assert(round(ext.r1_samples[0, 0, 0]) == 0)
    assert(event.r0.tels_with_data == {0})
    assert(event.r0.tel[0].adc_samples is None)
    assert(event.r1.tel[0].pe_samples[0, 0, 0] == ext.r1_samples[0, 0, 0])
