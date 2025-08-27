import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time

from ctapipe.core import ToolConfigurationError
from ctapipe.io import (
    HDF5EventSource,
    HDF5MonitoringSource,
    MonitoringTypes,
    get_hdf5_monitoring_types,
)
from ctapipe.io.hdf5dataformat import (
    DL0_TEL_POINTING_GROUP,
    DL1_CAMERA_COEFFICIENTS_GROUP,
    DL1_FLATFIELD_PEAK_TIME_GROUP,
    DL1_SKY_PEDESTAL_IMAGE_GROUP,
)


def test_subarray(calibpipe_camcalib_single_chunk):
    """test a simple subarray"""
    with HDF5MonitoringSource(input_url=calibpipe_camcalib_single_chunk) as source:
        assert source.subarray.telescope_types
        assert source.subarray.camera_types
        assert source.subarray.optics_types


def test_passing_subarray(dl1_file, calibpipe_camcalib_same_chunks):
    """test the functionality of passing a subarray from an EventSource"""
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
    """test the retrieval of monitoring types from HDF5 files"""
    # Test with a file that has no monitoring types
    with pytest.warns(UserWarning, match="No monitoring types found in"):
        no_monitoring_types = get_hdf5_monitoring_types(gamma_diffuse_full_reco_file)
        assert tuple([]) == no_monitoring_types
    # Test with a file that has pointing-related monitoring types
    assert tuple([MonitoringTypes.TELESCOPE_POINTINGS]) == get_hdf5_monitoring_types(
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
            MonitoringTypes.TELESCOPE_POINTINGS,
        ]
    ) == get_hdf5_monitoring_types(dl1_merged_monitoring_file)


def test_camcalib_filling(gamma_diffuse_full_reco_file, dl1_merged_monitoring_file):
    """test the monitoring filling for the camera calibration coefficients"""
    from ctapipe.io import read_table

    tel_id = 1
    # Read the camera monitoring data with the coefficients
    camcalib_coefficients = read_table(
        dl1_merged_monitoring_file,
        f"{DL1_CAMERA_COEFFICIENTS_GROUP}/tel_{tel_id:03d}",
    )[0]
    allowed_tels = {tel_id}
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
            # Check that the values match after filling the container
            for column in ["factor", "pedestal_offset", "time_shift", "outlier_mask"]:
                np.testing.assert_array_equal(
                    e.monitoring.tel[tel_id].camera.coefficients[column],
                    camcalib_coefficients[column],
                    err_msg=(
                        f"'{column}' do not match after reading the monitoring file "
                        "through the HDF5MonitoringSource."
                    ),
                )
        # Close the monitoring source
        monitoring_source.close()


def test_get_camera_monitoring_container_sims(calibpipe_camcalib_same_chunks):
    """test the get_camera_monitoring_container method with the monitoring source of simulation"""
    from ctapipe.io import read_table

    tel_id = 1
    # Read the camera monitoring data with the coefficients
    camcalib_coefficients = read_table(
        calibpipe_camcalib_same_chunks,
        f"{DL1_CAMERA_COEFFICIENTS_GROUP}/tel_{tel_id:03d}",
    )
    sky_pedestal_image = read_table(
        calibpipe_camcalib_same_chunks,
        f"{DL1_SKY_PEDESTAL_IMAGE_GROUP}/tel_{tel_id:03d}",
    )
    # Set outliers to NaNs
    for col in ["mean", "median", "std"]:
        sky_pedestal_image[col][sky_pedestal_image["outlier_mask"].data] = np.nan
    with HDF5MonitoringSource(
        input_url=calibpipe_camcalib_same_chunks
    ) as monitoring_source:
        cam_mon_con = monitoring_source.get_camera_monitoring_container(
            tel_id,
        )
        # Validate the returned container
        cam_mon_con.validate()
        for column in [
            "mean",
            "median",
            "std",
        ]:
            np.testing.assert_array_equal(
                cam_mon_con.pixel_statistics.sky_pedestal_image[column],
                sky_pedestal_image[column][0],
                err_msg=(
                    f"'{column}' do not match after reading the monitoring file "
                    "through the HDF5MonitoringSource for the sky pedestal image."
                ),
            )
        for column in [
            "factor",
            "pedestal_offset",
            "time_shift",
            "outlier_mask",
        ]:
            np.testing.assert_array_equal(
                cam_mon_con.coefficients[column],
                camcalib_coefficients[column][0],
                err_msg=(
                    f"'{column}' do not match after reading the monitoring file "
                    "through the HDF5MonitoringSource for the camera calibration."
                ),
            )
        # Check exceptions when requesting the camera monitoring container with unique
        # timestamps in the monitoring source of simulation
        t_start = monitoring_source.pixel_statistics[tel_id]["flatfield_image"][
            "time_start"
        ][0]
        unique_timestamps = Time(
            [t_start + 1.2 * u.s, t_start + 1.7 * u.s, t_start + 20 * u.day]
        )
        with pytest.warns(
            UserWarning,
            match="The function argument 'time' is provided, but the monitoring source is of simulated data.",
        ):
            monitoring_source.get_camera_monitoring_container(tel_id, unique_timestamps)


def test_get_camera_monitoring_container_obs(calibpipe_camcalib_same_chunks_obs):
    """test the get_camera_monitoring_container method with the monitoring source of observation"""
    from ctapipe.io import read_table

    tel_id = 1
    # Read the camera monitoring data with the coefficients
    camcalib_coefficients = read_table(
        calibpipe_camcalib_same_chunks_obs,
        f"{DL1_CAMERA_COEFFICIENTS_GROUP}/tel_{tel_id:03d}",
    )
    flatfield_peak_time = read_table(
        calibpipe_camcalib_same_chunks_obs,
        f"{DL1_FLATFIELD_PEAK_TIME_GROUP}/tel_{tel_id:03d}",
    )
    with HDF5MonitoringSource(
        input_url=calibpipe_camcalib_same_chunks_obs
    ) as monitoring_source:
        with pytest.raises(
            ValueError,
            match="Function argument 'time' must be provided for monitoring data from real observations.",
        ):
            monitoring_source.get_camera_monitoring_container(
                tel_id,
            )
        # Read start and end times from the flatfield image container
        t_start = monitoring_source.pixel_statistics[tel_id]["flatfield_image"][
            "time_start"
        ][0]
        t_end = monitoring_source.pixel_statistics[tel_id]["flatfield_image"][
            "time_end"
        ][-1]
        # Set the unique timestamp
        unique_timestamp = t_start + 0.2 * u.s
        # Get the camera monitoring container for the given unique timestamps
        cam_mon_con = monitoring_source.get_camera_monitoring_container(
            tel_id, unique_timestamp
        )
        # Validate the returned container
        cam_mon_con.validate()
        for column in [
            "factor",
            "pedestal_offset",
            "time_shift",
            "outlier_mask",
        ]:
            np.testing.assert_array_equal(
                cam_mon_con.coefficients[column],
                camcalib_coefficients[column][0],
                err_msg=(
                    f"'{column}' do not match after reading the monitoring file "
                    "through the HDF5MonitoringSource for the camera calibration."
                ),
            )
        for column in [
            "mean",
            "median",
            "std",
        ]:
            np.testing.assert_array_equal(
                cam_mon_con.pixel_statistics.flatfield_peak_time[column],
                flatfield_peak_time[column][0],
                err_msg=(
                    f"'{column}' do not match after reading the monitoring file "
                    "through the HDF5MonitoringSource for the flatfield peak time."
                ),
            )
        # Set the unique timestamps within the validity range
        unique_timestamps = Time([t_start + 0.2 * u.s, t_end - 0.2 * u.s])
        # Get the camera monitoring container for the given unique timestamps
        cam_mon_con = monitoring_source.get_camera_monitoring_container(
            tel_id, unique_timestamps
        )
        # Validate the returned container
        cam_mon_con.validate()

        # Check that the first coefficients match the expected values (first entry)
        for column in [
            "factor",
            "pedestal_offset",
            "time_shift",
            "outlier_mask",
        ]:
            np.testing.assert_array_equal(
                cam_mon_con.coefficients[column][0],
                camcalib_coefficients[column][0],
                err_msg=(
                    f"'{column}' do not match after reading the monitoring file "
                    "through the HDF5MonitoringSource for the camera calibration."
                ),
            )
        # Check that the second coefficients match the expected values (last entry)
        for column in [
            "factor",
            "pedestal_offset",
            "time_shift",
            "outlier_mask",
        ]:
            np.testing.assert_array_equal(
                cam_mon_con.coefficients[column][1],
                camcalib_coefficients[column][-1],
                err_msg=(
                    f"'{column}' do not match after reading the monitoring file "
                    "through the HDF5MonitoringSource for the camera calibration."
                ),
            )


def test_tel_pointing_filling(gamma_diffuse_full_reco_file, dl1_mon_pointing_file):
    """test the monitoring filling for the telescope pointings"""
    from ctapipe.io import read_table

    tel_id = 1
    # Read the camera monitoring data with the coefficients
    pointing_time = (
        read_table(
            dl1_mon_pointing_file,
            f"{DL0_TEL_POINTING_GROUP}/tel_{tel_id:03d}",
        )["time"][0]
        + 1 * u.s
    )
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


def test_camcalib_obs(gamma_diffuse_full_reco_file, calibpipe_camcalib_same_chunks_obs):
    """test the HDF5MonitoringSource with camcalib monitoring files from 'observation'"""
    from ctapipe.io import read_table

    tel_id = 1
    # Read the camera monitoring data with the coefficients
    camcalib_coefficients = read_table(
        calibpipe_camcalib_same_chunks_obs,
        f"{DL1_CAMERA_COEFFICIENTS_GROUP}/tel_{tel_id:03d}",
    )
    # Define some usual trigger times
    # Before the validity range should raise an exception
    trigger_time_before = camcalib_coefficients["time"][0] - 0.5 * u.s
    # Inside of the validity range should work smoothly
    # and values should match to the fifth entry
    trigger_time_middle = camcalib_coefficients["time"][5] + 0.5 * u.s
    # After the last validity start of a chunk should also work
    # and match the last entry.
    trigger_time_after = camcalib_coefficients["time"][-1] + 0.5 * u.s
    allowed_tels = {tel_id}
    with HDF5EventSource(
        input_url=gamma_diffuse_full_reco_file, allowed_tels=allowed_tels, max_events=1
    ) as source:
        monitoring_source = HDF5MonitoringSource(
            subarray=source.subarray,
            input_url=calibpipe_camcalib_same_chunks_obs,
            overwrite_telescope_pointings=False,
        )
        assert not monitoring_source.is_simulation
        assert monitoring_source.pixel_statistics
        assert monitoring_source.camera_coefficients
        assert not monitoring_source.telescope_pointings
        # Check that the camcalib_coefficients match the event calibration data
        for e in source:
            # Test exception of interpolating outside the valid range
            with pytest.raises(ValueError, match="Out of bounds: Requested timestamp"):
                e.trigger.time = trigger_time_before
                monitoring_source.fill_monitoring_container(e)

            for chunk_bin, trigger_time in {
                5: trigger_time_middle,
                -1: trigger_time_after,
            }.items():
                # Set the trigger time to the pointing time
                e.trigger.time = trigger_time
                # Fill the monitoring container for the event
                monitoring_source.fill_monitoring_container(e)
                # Check that the values match after filling the container
                for column in [
                    "factor",
                    "pedestal_offset",
                    "time_shift",
                    "outlier_mask",
                ]:
                    np.testing.assert_array_equal(
                        e.monitoring.tel[tel_id].camera.coefficients[column],
                        camcalib_coefficients[column][chunk_bin],
                        err_msg=(
                            f"'{column}' do not match after reading the monitoring file "
                            "through the HDF5MonitoringSource."
                        ),
                    )
        # Close the monitoring source
        monitoring_source.close()


def test_common_exceptions(
    gamma_diffuse_full_reco_file, calibpipe_camcalib_same_chunks
):
    """test the common exceptions of the HDF5MonitoringSource"""
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
