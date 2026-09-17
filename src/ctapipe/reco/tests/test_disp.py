from contextlib import ExitStack

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from ctapipe.containers import CoordinateFrameType
from ctapipe.reco.disp import compute_true_disp
from ctapipe.reco.preprocessing import telescope_to_horizontal


def make_disp_table(subarray_pointing):
    pointing_alt = np.full(3, 70) * u.deg
    pointing_az = np.full(3, 20) * u.deg
    fov_lon = np.array([1, 0, -1]) * u.deg
    fov_lat = np.array([0, 2, 0]) * u.deg
    true_alt, true_az = telescope_to_horizontal(
        lon=fov_lon,
        lat=fov_lat,
        pointing_alt=pointing_alt,
        pointing_az=pointing_az,
    )

    table = Table(
        {
            "obs_id": [1, 1, 1],
            "event_id": [1, 2, 3],
            "tel_id": [1, 1, 1],
            "true_alt": true_alt,
            "true_az": true_az,
            "hillas_psi": [0, 90, 0] * u.deg,
            "hillas_fov_lon": [0.25, -1.0, 0.5] * u.deg,
            "hillas_fov_lat": [-0.5, 0.5, 1.0] * u.deg,
        }
    )

    if subarray_pointing:
        table["subarray_pointing_frame"] = np.int8(CoordinateFrameType.ALTAZ.value)
        table["subarray_pointing_lat"] = pointing_alt
        table["subarray_pointing_lon"] = pointing_az
    else:
        table["telescope_pointing_altitude"] = pointing_alt
        table["telescope_pointing_azimuth"] = pointing_az

    return table


@pytest.mark.parametrize("subarray_pointing", [False, True])
def test_compute_true_disp(subarray_pointing):
    table = make_disp_table(subarray_pointing=subarray_pointing)

    with ExitStack() as stack:
        # we expect a warning, but only in the subarray_pointing case
        if subarray_pointing:
            msg = "falling back to array pointing"
            stack.enter_context(pytest.warns(UserWarning, match=msg))

        disp = compute_true_disp(table)

    np.testing.assert_allclose(disp.to_value(u.deg), [0.75, 1.5, -1.5])


def test_compute_true_angular_error():
    from astropy.coordinates import angular_separation

    from ctapipe.reco.disp import compute_true_angular_error

    reco_alt = [70, 70, 20] * u.deg
    reco_az = [0, 0, 0] * u.deg
    true_alt = [70, 70, 20] * u.deg
    true_az = [0, 10, 180] * u.deg

    error = compute_true_angular_error(reco_alt, reco_az, true_alt, true_az)

    # matches astropy's great-circle angular separation
    expected = angular_separation(reco_az, reco_alt, true_az, true_alt)
    np.testing.assert_allclose(error.to_value(u.deg), expected.to_value(u.deg))

    # zero error when reconstructed == true direction
    assert error[0].to_value(u.deg) == pytest.approx(0.0, abs=1e-9)

    # a 10 deg azimuth offset at 70 deg altitude is a much smaller on-sky
    # separation than 10 deg: the small-angle approximation would be wrong
    assert error[1].to_value(u.deg) == pytest.approx(3.42, abs=0.01)

    # large separations are handled correctly (no small-angle approximation)
    assert error[2].to_value(u.deg) == pytest.approx(140.0, abs=1e-6)
