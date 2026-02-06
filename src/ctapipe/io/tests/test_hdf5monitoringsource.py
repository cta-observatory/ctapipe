import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time

from ctapipe.exceptions import InputMissing
from ctapipe.io import (
    EventSource,
    HDF5MonitoringSource,
    MonitoringType,
    get_hdf5_monitoring_types,
    read_table,
)
from ctapipe.io.hdf5dataformat import (
    DL0_TEL_POINTING_GROUP,
    DL1_CAMERA_COEFFICIENTS_GROUP,
    DL1_FLATFIELD_PEAK_TIME_GROUP,
    DL1_SKY_PEDESTAL_IMAGE_GROUP,
)
from ctapipe.utils import get_dataset_path


def test_hdf5_monitoring_source_subarray():
    """test a simple subarray"""
    file = get_dataset_path("calibpipe_camcalib_sims_single_chunk_i0.2.0.dl1.h5")
    with HDF5MonitoringSource(input_files=[file]) as source:
        assert source.subarray.telescope_types
        assert source.subarray.camera_types
        assert source.subarray.optics_types


def test_passing_subarray(dl1_file, calibpipe_camcalib_obslike_same_chunks):
    """test the functionality of passing a subarray from an EventSource"""
    allowed_tels = {1}
    with EventSource(input_url=dl1_file, allowed_tels=allowed_tels) as source:
        monitoring_source = HDF5MonitoringSource(
            subarray=source.subarray,
            input_files=[calibpipe_camcalib_obslike_same_chunks],
        )
        assert monitoring_source.subarray.tel_ids == source.subarray.tel_ids


def test_get_monitoring_types(
    proton_dl2_train_small_h5,
    dl1_mon_pointing_file,
    calibpipe_camcalib_obslike_different_chunks,
    dl1_merged_monitoring_file,
):
    """test the retrieval of monitoring types from HDF5 files"""
    # Test with a file that has no monitoring types
    with pytest.warns(UserWarning, match="No monitoring types found in"):
        no_monitoring_types = get_hdf5_monitoring_types(proton_dl2_train_small_h5)
        assert tuple([]) == no_monitoring_types
    # Test with a file that has pointing-related monitoring types
    assert tuple([MonitoringType.TELESCOPE_POINTINGS]) == get_hdf5_monitoring_types(
        dl1_mon_pointing_file
    )
    # Test with a file that has camera-related monitoring types
    assert tuple(
        [MonitoringType.PIXEL_STATISTICS, MonitoringType.CAMERA_COEFFICIENTS]
    ) == get_hdf5_monitoring_types(calibpipe_camcalib_obslike_different_chunks)
    # Test with a file that has all current monitoring types
    assert tuple(
        [
            MonitoringType.PIXEL_STATISTICS,
            MonitoringType.CAMERA_COEFFICIENTS,
            MonitoringType.TELESCOPE_POINTINGS,
        ]
    ) == get_hdf5_monitoring_types(dl1_merged_monitoring_file)


def test_camcalib_filling(prod6_gamma_simtel_path, dl1_merged_monitoring_file):
    """test the monitoring filling for the camera calibration coefficients"""

    tel_id = 1
    # Read the camera monitoring data with the coefficients
    camcalib_coefficients = read_table(
        dl1_merged_monitoring_file,
        f"{DL1_CAMERA_COEFFICIENTS_GROUP}/tel_{tel_id:03d}",
    )[0]
    allowed_tels = {tel_id}
    with EventSource(
        input_url=prod6_gamma_simtel_path, allowed_tels=allowed_tels, max_events=1
    ) as source:
        monitoring_source = HDF5MonitoringSource(
            subarray=source.subarray,
            input_files=[dl1_merged_monitoring_file],
        )
        assert monitoring_source.is_simulation
        assert monitoring_source.pixel_statistics
        assert monitoring_source.camera_coefficients
        assert monitoring_source.telescope_pointings
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


def test_get_camera_monitoring_container_sims(calibpipe_camcalib_sims_single_chunk):
    """test the get_camera_monitoring_container method with the monitoring source of simulation"""

    tel_id = 1
    # Read the camera monitoring data with the coefficients
    camcalib_coefficients = read_table(
        calibpipe_camcalib_sims_single_chunk,
        f"{DL1_CAMERA_COEFFICIENTS_GROUP}/tel_{tel_id:03d}",
    )
    sky_pedestal_image = read_table(
        calibpipe_camcalib_sims_single_chunk,
        f"{DL1_SKY_PEDESTAL_IMAGE_GROUP}/tel_{tel_id:03d}",
    )
    # Set outliers to NaNs
    for col in ["mean", "median", "std"]:
        sky_pedestal_image[col][sky_pedestal_image["outlier_mask"].data] = np.nan
    with HDF5MonitoringSource(
        input_files=[calibpipe_camcalib_sims_single_chunk]
    ) as monitoring_source:
        camera_mon_con = monitoring_source.get_camera_monitoring_container(
            tel_id,
        )
        # Validate the returned container
        camera_mon_con.validate()
        for column in [
            "mean",
            "median",
            "std",
        ]:
            np.testing.assert_array_equal(
                camera_mon_con.pixel_statistics.pedestal_image[column],
                sky_pedestal_image[column][0],
                err_msg=(
                    f"'{column}' do not match after reading the monitoring file "
                    "through the HDF5MonitoringSource for the pedestal image."
                ),
            )
        for column in [
            "factor",
            "pedestal_offset",
            "time_shift",
            "outlier_mask",
        ]:
            np.testing.assert_array_equal(
                camera_mon_con.coefficients[column],
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


def test_get_camera_monitoring_container_obs(calibpipe_camcalib_obslike_same_chunks):
    """test the get_camera_monitoring_container method with the monitoring source of observation"""

    tel_id = 1
    # Read the camera monitoring data with the coefficients
    camcalib_coefficients = read_table(
        calibpipe_camcalib_obslike_same_chunks,
        f"{DL1_CAMERA_COEFFICIENTS_GROUP}/tel_{tel_id:03d}",
    )
    flatfield_peak_time = read_table(
        calibpipe_camcalib_obslike_same_chunks,
        f"{DL1_FLATFIELD_PEAK_TIME_GROUP}/tel_{tel_id:03d}",
    )
    with HDF5MonitoringSource(
        subarray=None,
        input_files=[calibpipe_camcalib_obslike_same_chunks],
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
        unique_timestamp = t_start - 0.2 * u.s
        # Test exception of interpolating outside the valid range
        with pytest.raises(
            ValueError,
            match="Out of bounds: Requested timestamp",
        ):
            monitoring_source.get_camera_monitoring_container(
                tel_id, unique_timestamp, timestamp_tolerance=0.1 * u.s
            )
        # Get the camera monitoring container for the given unique timestamps
        camera_mon_con = monitoring_source.get_camera_monitoring_container(
            tel_id, unique_timestamp, timestamp_tolerance=0.25 * u.s
        )
        # Validate the returned container
        camera_mon_con.validate()
        for column in [
            "factor",
            "pedestal_offset",
            "time_shift",
            "outlier_mask",
        ]:
            np.testing.assert_array_equal(
                camera_mon_con.coefficients[column],
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
                camera_mon_con.pixel_statistics.flatfield_peak_time[column],
                flatfield_peak_time[column][0],
                err_msg=(
                    f"'{column}' do not match after reading the monitoring file "
                    "through the HDF5MonitoringSource for the flatfield peak time."
                ),
            )
        # Set the unique timestamps within the validity range
        unique_timestamps = Time([t_start + 0.2 * u.s, t_end + 0.2 * u.s])
        # Get the camera monitoring container for the given unique timestamps
        camera_mon_con = monitoring_source.get_camera_monitoring_container(
            tel_id, unique_timestamps, timestamp_tolerance=0.25 * u.s
        )
        # Validate the returned container
        camera_mon_con.validate()

        # Check that the first coefficients match the expected values (first entry)
        for column in [
            "factor",
            "pedestal_offset",
            "time_shift",
            "outlier_mask",
        ]:
            np.testing.assert_array_equal(
                camera_mon_con.coefficients[column][0],
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
                camera_mon_con.coefficients[column][1],
                camcalib_coefficients[column][-1],
                err_msg=(
                    f"'{column}' do not match after reading the monitoring file "
                    "through the HDF5MonitoringSource for the camera calibration."
                ),
            )


def test_tel_pointing_filling(prod6_gamma_simtel_path, dl1_merged_monitoring_file_obs):
    """test the monitoring filling for the telescope pointings"""

    tel_id = 1
    # Read the camera monitoring data with the coefficients
    pointing_time = (
        read_table(
            dl1_merged_monitoring_file_obs,
            f"{DL0_TEL_POINTING_GROUP}/tel_{tel_id:03d}",
        )["time"][0]
        + 1 * u.s
    )
    allowed_tels = {tel_id}
    with EventSource(
        input_url=prod6_gamma_simtel_path, allowed_tels=allowed_tels, max_events=1
    ) as source:
        monitoring_source = HDF5MonitoringSource(
            subarray=source.subarray,
            input_files=[dl1_merged_monitoring_file_obs],
        )
        assert not monitoring_source.is_simulation
        assert monitoring_source.pixel_statistics
        assert monitoring_source.camera_coefficients
        assert monitoring_source.telescope_pointings
        for e in source:
            # Test exception of interpolating outside the valid range
            with pytest.raises(ValueError, match="Out of bounds: Requested timestamp"):
                monitoring_source.fill_monitoring_container(e)
            # Set the trigger time to the pointing time
            e.trigger.time = pointing_time
            # Set pointing to different values to ensure they get overwritten
            e.monitoring.tel[tel_id].pointing.azimuth = 90.0 * u.deg
            e.monitoring.tel[tel_id].pointing.altitude = 45.0 * u.deg
            old_az = e.monitoring.tel[tel_id].pointing.azimuth
            old_alt = e.monitoring.tel[tel_id].pointing.altitude
            # Fill the monitoring container for the event and overwrite the pointing
            monitoring_source.fill_monitoring_container(e)
            # Check that the values changed
            assert not u.isclose(e.monitoring.tel[tel_id].pointing.azimuth, old_az)
            assert not u.isclose(e.monitoring.tel[tel_id].pointing.altitude, old_alt)


def test_camcalib_obs(prod6_gamma_simtel_path, calibpipe_camcalib_obslike_same_chunks):
    """test the HDF5MonitoringSource with camcalib monitoring files from 'observation'"""

    tel_id = 1
    # Read the camera monitoring data with the coefficients
    camcalib_coefficients = read_table(
        calibpipe_camcalib_obslike_same_chunks,
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
    with EventSource(
        input_url=prod6_gamma_simtel_path, allowed_tels=allowed_tels, max_events=1
    ) as source:
        monitoring_source = HDF5MonitoringSource(
            subarray=source.subarray,
            input_files=[calibpipe_camcalib_obslike_same_chunks],
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


def test_hdf5_monitoring_source_multi_files_loading(
    dl1_mon_pointing_file_obs, calibpipe_camcalib_obslike_same_chunks
):
    """Test loading multiple HDF5 monitoring files in the HDF5MonitoringSource"""

    tel_id = 1
    # Read the camera monitoring data with the coefficients for the timestamp
    camcalib_coefficients = read_table(
        calibpipe_camcalib_obslike_same_chunks,
        f"{DL1_CAMERA_COEFFICIENTS_GROUP}/tel_{tel_id:03d}",
    )
    monitoring_source = HDF5MonitoringSource(
        input_files=[calibpipe_camcalib_obslike_same_chunks, dl1_mon_pointing_file_obs],
    )
    assert not monitoring_source.is_simulation
    assert monitoring_source.pixel_statistics
    assert monitoring_source.camera_coefficients
    assert monitoring_source.telescope_pointings
    # Check that we can retrieve the camera monitoring container
    camera_mon_con = monitoring_source.get_camera_monitoring_container(
        tel_id, camcalib_coefficients["time"][0] + 0.5 * u.s
    )
    # Validate the returned container
    camera_mon_con.validate()
    # Check that the first coefficients match the expected values (first entry)
    for column in [
        "factor",
        "pedestal_offset",
        "time_shift",
        "outlier_mask",
    ]:
        np.testing.assert_array_equal(
            camera_mon_con.coefficients[column],
            camcalib_coefficients[column][0],
            err_msg=(
                f"'{column}' do not match after reading the monitoring file "
                "through the HDF5MonitoringSource for the camera calibration."
            ),
        )
    # Check that we can retrieve the telescope pointing container
    tel_pointing_mon_con = monitoring_source.get_telescope_pointing_container(
        tel_id, camcalib_coefficients["time"][0] + 0.5 * u.s
    )
    assert tel_pointing_mon_con["altitude"] is not None
    assert tel_pointing_mon_con["azimuth"] is not None


def test_hdf5_monitoring_source_exceptions_and_warnings(
    prod6_gamma_simtel_path,
    calibpipe_camcalib_sims_single_chunk,
    calibpipe_camcalib_obslike_same_chunks,
    dl1_mon_pointing_file,
):
    """test the common exceptions and warnings of the HDF5MonitoringSource"""
    # Pass a subarray with more telescopes than available in the monitoring file.
    # This should raise an IOError.
    with EventSource(
        input_url=prod6_gamma_simtel_path, allowed_tels={2, 3, 4}
    ) as source:
        with pytest.raises(
            IOError, match="Incompatible subarray descriptions found in input files."
        ):
            HDF5MonitoringSource(
                subarray=source.subarray,
                input_files=[calibpipe_camcalib_sims_single_chunk],
            )
    # Do not provide an input file. This should raise an InputMissing.
    with pytest.raises(
        InputMissing,
        match="No input files provided",
    ):
        HDF5MonitoringSource()
    # Do not provide an input files with inconsistent simulation flags. This should raise an IOError.
    with pytest.raises(
        IOError, match="HDF5MonitoringSource: Inconsistent simulation flags found in"
    ):
        HDF5MonitoringSource(
            input_files=[dl1_mon_pointing_file, calibpipe_camcalib_obslike_same_chunks]
        )
    # Test that we can open a file with pointing data (even for simulation)
    monitoring_source = HDF5MonitoringSource(
        subarray=None,
        input_files=[dl1_mon_pointing_file],
    )
    assert monitoring_source.has_pointings
    # Warns if overlapping monitoring types are found in multiple input files.
    with pytest.warns(
        UserWarning,
        match="File ",
    ):
        HDF5MonitoringSource(
            subarray=None,
            input_files=[
                calibpipe_camcalib_obslike_same_chunks,
                calibpipe_camcalib_obslike_same_chunks,
            ],
        )


def test_get_table(calibpipe_camcalib_sims_single_chunk):
    """test the get_table method"""
    tel_id = 1

    with HDF5MonitoringSource(
        input_files=[calibpipe_camcalib_sims_single_chunk]
    ) as source:
        # Test getting camera coefficients table
        coeffs_table = source.get_table(
            MonitoringType.CAMERA_COEFFICIENTS, tel_id=tel_id
        )
        assert coeffs_table is not None
        assert "time" in coeffs_table.colnames
        assert "factor" in coeffs_table.colnames
        assert "pedestal_offset" in coeffs_table.colnames
        assert len(coeffs_table) > 0

        # Test getting pixel statistics with subtype
        flatfield_table = source.get_table(
            MonitoringType.PIXEL_STATISTICS, tel_id=tel_id, subtype="flatfield_image"
        )
        assert flatfield_table is not None
        assert "mean" in flatfield_table.colnames
        assert "median" in flatfield_table.colnames
        assert "std" in flatfield_table.colnames

        # Test error when monitoring type not available
        with pytest.raises(KeyError, match="not available"):
            source.get_table(MonitoringType.TELESCOPE_POINTINGS, tel_id=tel_id)

        # Test error when tel_id missing for telescope-level data
        with pytest.raises(TypeError, match="tel_id is required"):
            source.get_table(MonitoringType.CAMERA_COEFFICIENTS)

        # Test error when subtype missing for pixel statistics
        with pytest.raises(KeyError, match="subtype parameter is required"):
            source.get_table(MonitoringType.PIXEL_STATISTICS, tel_id=tel_id)

        # Test error for invalid subtype
        with pytest.raises(KeyError, match="Unknown subtype"):
            source.get_table(
                MonitoringType.PIXEL_STATISTICS,
                tel_id=tel_id,
                subtype="invalid_subtype",
            )


def test_get_table_pointing(dl1_merged_monitoring_file):
    """test get_table method for telescope pointing"""
    tel_id = 1

    with HDF5MonitoringSource(input_files=[dl1_merged_monitoring_file]) as source:
        # Test getting telescope pointing table
        pointing_table = source.get_table(
            MonitoringType.TELESCOPE_POINTINGS, tel_id=tel_id
        )
        assert pointing_table is not None
        assert "time" in pointing_table.colnames
        assert "azimuth" in pointing_table.colnames
        assert "altitude" in pointing_table.colnames


def test_get_values_camera_coefficients(calibpipe_camcalib_sims_single_chunk):
    """test the get_values method for camera coefficients"""
    tel_id = 1

    with HDF5MonitoringSource(
        input_files=[calibpipe_camcalib_sims_single_chunk]
    ) as source:
        # For simulation, time can be None (uses first entry)
        values = source.get_values(
            MonitoringType.CAMERA_COEFFICIENTS, time=None, tel_id=tel_id
        )
        assert isinstance(values, dict)
        assert "factor" in values
        assert "pedestal_offset" in values
        assert "time_shift" in values
        assert "outlier_mask" in values
        assert "is_valid" in values

        # Test with explicit time
        table = source.get_table(MonitoringType.CAMERA_COEFFICIENTS, tel_id=tel_id)
        time = Time(table["time"][0], format="mjd")
        values = source.get_values(
            MonitoringType.CAMERA_COEFFICIENTS, time=time, tel_id=tel_id
        )
        assert isinstance(values, dict)


def test_get_values_pixel_statistics(calibpipe_camcalib_sims_single_chunk):
    """test the get_values method for pixel statistics"""
    tel_id = 1

    with HDF5MonitoringSource(
        input_files=[calibpipe_camcalib_sims_single_chunk]
    ) as source:
        # Test with subtype
        values = source.get_values(
            MonitoringType.PIXEL_STATISTICS,
            time=None,
            tel_id=tel_id,
            subtype="flatfield_image",
        )
        assert isinstance(values, dict)
        assert "mean" in values
        assert "median" in values
        assert "std" in values

        # Test error when subtype missing
        with pytest.raises(KeyError, match="subtype parameter is required"):
            source.get_values(MonitoringType.PIXEL_STATISTICS, time=None, tel_id=tel_id)


def test_get_values_telescope_pointing(dl1_merged_monitoring_file):
    """test the get_values method for telescope pointing"""
    from astropy.coordinates import SkyCoord

    tel_id = 1

    with HDF5MonitoringSource(input_files=[dl1_merged_monitoring_file]) as source:
        # Get a time from the pointing table
        table = source.get_table(MonitoringType.TELESCOPE_POINTINGS, tel_id=tel_id)
        time = Time(table["time"][0], format="unix")

        # Get interpolated pointing
        pointing = source.get_values(
            MonitoringType.TELESCOPE_POINTINGS, time=time, tel_id=tel_id
        )
        assert isinstance(pointing, SkyCoord)
        assert pointing.alt.unit.is_equivalent(u.rad)
        assert pointing.az.unit.is_equivalent(u.rad)

        # Test with array of times
        times = Time(table["time"][:3], format="unix")
        pointings = source.get_values(
            MonitoringType.TELESCOPE_POINTINGS, time=times, tel_id=tel_id
        )
        assert isinstance(pointings, SkyCoord)
        assert len(pointings) == 3


def test_get_values_errors(calibpipe_camcalib_sims_single_chunk):
    """test error handling in get_values method"""
    tel_id = 1

    with HDF5MonitoringSource(
        input_files=[calibpipe_camcalib_sims_single_chunk]
    ) as source:
        # Test error when monitoring type not available
        with pytest.raises(KeyError, match="not available"):
            source.get_values(
                MonitoringType.TELESCOPE_POINTINGS, time=None, tel_id=tel_id
            )

        # Test error when tel_id missing
        with pytest.raises(TypeError, match="tel_id is required"):
            source.get_values(MonitoringType.CAMERA_COEFFICIENTS, time=None)
