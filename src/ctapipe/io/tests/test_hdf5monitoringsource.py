import numpy as np
import pytest
import astropy.units as u

from ctapipe.core import ToolConfigurationError
from ctapipe.io import (
    HDF5EventSource,
    HDF5MonitoringSource,
    MonitoringTypes,
    get_hdf5_monitoring_types,
)


def test_subarray(calibpipe_camcalib_single_chunk):
    with HDF5MonitoringSource(input_url=calibpipe_camcalib_single_chunk) as source:
        assert source.subarray.telescope_types
        assert source.subarray.camera_types
        assert source.subarray.optics_types


def test_passing_subarray(dl1_file, calibpipe_camcalib_same_chunks):
    allowed_tels = {1}
    with HDF5EventSource(input_url=dl1_file, allowed_tels=allowed_tels) as source:
        monitoring_source = HDF5MonitoringSource(
            subarray=source.subarray, input_url=calibpipe_camcalib_same_chunks
        )
        assert (
            monitoring_source.subarray.telescope_types
            == source.subarray.telescope_types
        )
        assert monitoring_source.subarray.camera_types == source.subarray.camera_types
        assert monitoring_source.subarray.optics_types == source.subarray.optics_types
        # Close the monitoring source
        monitoring_source.close()


def test_get_monitoring_types(
    gamma_diffuse_full_reco_file,
    dl1_mon_pointing_file,
    calibpipe_camcalib_different_chunks,
    dl1_merged_monitoring_file,
):
    # Test with a file that has no monitoring types
    assert tuple([]) == get_hdf5_monitoring_types(gamma_diffuse_full_reco_file)
    # Test with a file that has pointing-related monitoring types
    assert tuple([MonitoringTypes.TELESCOPE_POINTING]) == get_hdf5_monitoring_types(
        dl1_mon_pointing_file
    )
    # Test with a file that has camera-related monitoring types
    assert tuple(
        [MonitoringTypes.PIXEL_STATISTICS, MonitoringTypes.CAMERA_COEFFICIENTS]
    ) == get_hdf5_monitoring_types(calibpipe_camcalib_different_chunks)
    # Test with a file that has all current monitoring types
    assert tuple(
        [
            MonitoringTypes.PIXEL_STATISTICS,
            MonitoringTypes.CAMERA_COEFFICIENTS,
            MonitoringTypes.TELESCOPE_POINTING,
        ]
    ) == get_hdf5_monitoring_types(dl1_merged_monitoring_file)


def test_camcalib_filling(gamma_diffuse_full_reco_file, dl1_merged_monitoring_file):
    from ctapipe.io import read_table

    # Read the camera monitoring data with the coefficients
    camcalib_coefficients = read_table(
        dl1_merged_monitoring_file,
        "/dl1/monitoring/telescope/calibration/camera/coefficients/tel_001",
    )
    allowed_tels = {1}
    with HDF5EventSource(
        input_url=gamma_diffuse_full_reco_file, allowed_tels=allowed_tels, max_events=1
    ) as source:
        monitoring_source = HDF5MonitoringSource(
            subarray=source.subarray,
            input_url=dl1_merged_monitoring_file,
            overwrite_telescope_pointings=False,
        )
        assert monitoring_source.is_simulation
        assert monitoring_source.pixel_statistics
        assert monitoring_source.camera_coefficients
        assert not monitoring_source.telescope_pointings
        # Check that the camcalib_coefficients match the event calibration data
        for e in source:
            # Fill the monitoring container for the event
            monitoring_source.fill_monitoring_container(e)
            # Check that the values match
            np.testing.assert_array_equal(
                e.monitoring.tel[1].camera.coefficients["factor"],
                camcalib_coefficients["factor"][0],
                err_msg="Factors do not match after reading the monitoring file through the HDF5MonitoringSource",
            )
            np.testing.assert_array_equal(
                e.monitoring.tel[1].camera.coefficients["pedestal_offset"],
                camcalib_coefficients["pedestal_offset"][0],
                err_msg="Pedestal offsets do not match after reading the monitoring file through the HDF5MonitoringSource",
            )
            np.testing.assert_array_equal(
                e.monitoring.tel[1].camera.coefficients["time_shift"],
                camcalib_coefficients["time_shift"][0],
                err_msg="Time shifts do not match after reading the monitoring file through the HDF5MonitoringSource",
            )
            np.testing.assert_array_equal(
                e.monitoring.tel[1].camera.coefficients["outlier_mask"],
                camcalib_coefficients["outlier_mask"][0],
                err_msg="Outlier masks do not match after reading the monitoring file through the HDF5MonitoringSource",
            )
        # Close the monitoring source
        monitoring_source.close()


def test_tel_pointing_filling(gamma_diffuse_full_reco_file, dl1_mon_pointing_file):
    from ctapipe.io import read_table
    tel_id = 1
    # Read the camera monitoring data with the coefficients
    pointing_time = read_table(
        dl1_mon_pointing_file,
        f"/dl1/monitoring/telescope/calibration/pointing/tel_{tel_id:03d}",
    )["time"][0] + 1 * u.s
    allowed_tels = {tel_id}
    with HDF5EventSource(
        input_url=gamma_diffuse_full_reco_file, allowed_tels=allowed_tels, max_events=1
    ) as source:
        monitoring_source = HDF5MonitoringSource(
            subarray=source.subarray,
            input_url=dl1_mon_pointing_file,
            overwrite_telescope_pointings=True,
        )
        assert monitoring_source.is_simulation
        assert not monitoring_source.pixel_statistics
        assert not monitoring_source.camera_coefficients
        assert monitoring_source.telescope_pointings
        for e in source:
            # Test exception of interpolating outside the valid range
            with pytest.raises(ValueError, match="A value"):
                monitoring_source.fill_monitoring_container(e)
            # Set the trigger time to the pointing time
            e.trigger.time = pointing_time
            # Save the old pointing values
            old_az = e.monitoring.tel[tel_id].pointing.azimuth
            old_alt = e.monitoring.tel[tel_id].pointing.altitude
            # Fill the monitoring container for the event and overwrite the pointing
            monitoring_source.fill_monitoring_container(e)
            # Check that the values do not match
            assert e.monitoring.tel[tel_id].pointing.azimuth != old_az
            assert e.monitoring.tel[tel_id].pointing.altitude != old_alt
        # Close the monitoring source
        monitoring_source.close()

def test_exceptions(gamma_diffuse_full_reco_file, calibpipe_camcalib_same_chunks):
    # Pass a subarray with more telescopes than available in the monitoring file.
    # This should raise a ToolConfigurationError.
    with HDF5EventSource(input_url=gamma_diffuse_full_reco_file) as source:
        with pytest.raises(
            ToolConfigurationError, match="HDF5MonitoringSource: Requested telescopes"
        ):
            HDF5MonitoringSource(
                subarray=source.subarray, input_url=calibpipe_camcalib_same_chunks
            )
    # Do not pass a subarray and switch off the enforcement of a subarray description.
    # This should raise a NotImplementedError.
    with pytest.raises(NotImplementedError, match="Subarray is not defined"):
        HDF5MonitoringSource(
            input_url=calibpipe_camcalib_same_chunks,
            enforce_subarray_description=False,
        )
    # Request to overwrite the telescope pointings, but none are available.
    # This should raise a ToolConfigurationError.
    with pytest.raises(
        ToolConfigurationError,
        match="HDF5MonitoringSource: Telescope pointings are not available",
    ):
        HDF5MonitoringSource(
            input_url=calibpipe_camcalib_same_chunks,
            overwrite_telescope_pointings=True,
        )
