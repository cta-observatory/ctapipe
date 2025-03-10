"""Functions handling time, mainly conversion of CTA timestamps to astropy."""
import numpy as np
from astropy.time import Time

EPOCH = Time(0, format="unix_tai", scale="tai")
DAY_TO_S = 86400
CENTRAL_MODULE = 132
S_TO_QNS = 4e9
QNS_TO_S = 0.25e-9


def cta_high_res_to_time(seconds, quarter_nanoseconds):
    """Convert cta high resolution timestamp to astropy Time."""
    # unix_tai accepts two floats for maximum precision
    # we can just pass integral and fractional part
    fractional_seconds = quarter_nanoseconds * QNS_TO_S
    return Time(
        seconds,
        fractional_seconds,
        format="unix_tai",
        # this is only for displaying iso timestamp, not any actual precision
        precision=9,
    )


def to_seconds(days):
    """Return whole and fractional seconds from a number in days."""
    seconds = days * DAY_TO_S
    return np.divmod(seconds, 1)


def time_to_cta_high_res(time):
    """Convert astropy Time to cta high precision timestamp."""
    # make sure we are in TAI
    time = time.tai

    # internally, astropy always uses jd values
    # jd1 is integral and jd2 is in [-0.5, 0.5]
    # we get the integral and fractional seconds from both jd values
    # relative to the epoch
    seconds_jd1, fractional_seconds_jd1 = to_seconds(time.jd1 - EPOCH.jd1)
    seconds_jd2, fractional_seconds_jd2 = to_seconds(time.jd2 - EPOCH.jd2)

    # add up the integral number of seconds
    seconds = seconds_jd1 + seconds_jd2

    # convert fractional seconds to quarter nanoseconds
    fractional_seconds = fractional_seconds_jd1 + fractional_seconds_jd2
    quarter_nanoseconds = np.round(fractional_seconds * S_TO_QNS)

    # convert to uint32
    seconds = np.asanyarray(seconds, dtype=np.uint32)
    quarter_nanoseconds = np.asanyarray(quarter_nanoseconds, dtype=np.uint32)
    return seconds, quarter_nanoseconds
