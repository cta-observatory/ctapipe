import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time

from ctapipe.containers import NAN_TIME

roundtrip_times = [
    "2020-01-01T00:00:00.12345678925",
    "2020-01-01T00:00:00.1234567895",
    "2020-01-01T00:00:00.12345678975",
    "2020-01-01T00:00:00.123456790",
]


@pytest.mark.parametrize("timestamp", roundtrip_times)
def test_ctao_high_round_trip(timestamp):
    from ctapipe.time import ctao_high_res_to_time, time_to_ctao_high_res

    # note: precision=9 only affects text conversion, not actual precision
    time = Time(timestamp, scale="tai", precision=9)
    time_s, time_qns = time_to_ctao_high_res(time)
    time_back = ctao_high_res_to_time(time_s, time_qns)

    roundtrip_error = (time - time_back).to_value(u.ns)
    np.testing.assert_almost_equal(roundtrip_error, 0.0)


test_data = [
    (Time(0, 12.25e-9, format="unix_tai"), 0, 49),
    (Time(12345, 12.25e-9, format="unix_tai"), 12345, 49),
    (Time(65123, 123456.25e-9, format="unix_tai"), 65123, 493825),
    (Time("2200-01-02T00:00:00"), np.iinfo(np.uint32).max, 0),
    (NAN_TIME, 0, 0),
]


@pytest.mark.parametrize(("time", "expected_s", "expected_qns"), test_data)
def test_time_to_ctao_high_res(time, expected_s, expected_qns):
    from ctapipe.time import time_to_ctao_high_res

    time_s, time_qns = time_to_ctao_high_res(time)
    assert time_s == expected_s
    assert time_qns == expected_qns
