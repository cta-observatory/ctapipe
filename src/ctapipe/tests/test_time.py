import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time

roundtrip_times = [
    "2020-01-01T00:00:00.12345678925",
    "2020-01-01T00:00:00.1234567895",
    "2020-01-01T00:00:00.12345678975",
    "2020-01-01T00:00:00.123456790",
]


@pytest.mark.parametrize("timestamp", roundtrip_times)
def test_cta_high_round_trip(timestamp):
    from ctapipe_io_zfits.time import cta_high_res_to_time, time_to_cta_high_res

    # note: precision=9 only affects text conversion, not actual precision
    time = Time(timestamp, scale="tai", precision=9)
    seconds, quarter_nanoseconds = time_to_cta_high_res(time)
    time_back = cta_high_res_to_time(seconds, quarter_nanoseconds)

    roundtrip_error = (time - time_back).to_value(u.ns)
    np.testing.assert_almost_equal(roundtrip_error, 0.0)


test_data = [
    (Time(0, 12.25e-9, format="unix_tai"), 0, 49),
    (Time(12345, 12.25e-9, format="unix_tai"), 12345, 49),
]


@pytest.mark.parametrize(("time", "expected_s", "expected_qns"), test_data)
def test_cta_time_to_cta_high_res(time, expected_s, expected_qns):
    from ctapipe_io_zfits.time import time_to_cta_high_res

    seconds, quarter_nanoseconds = time_to_cta_high_res(time)
    assert seconds == expected_s
    assert quarter_nanoseconds == expected_qns
