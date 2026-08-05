import astropy.units as u
import pytest

from ctapipe.calib.optics.calibrator import PointingCalibrator
from ctapipe.containers import (
    ArrayEventContainer,
    TelescopeStructureDisplacementContainer,
    TelescopeStructurePointingContainer,
)


@pytest.fixture
def pointing_calibrator(example_subarray):
    return PointingCalibrator(subarray=example_subarray)


def test_apply_structure_displacement(pointing_calibrator):
    event = ArrayEventContainer()
    tel_id = 1
    event.trigger.tels_with_trigger = [tel_id]

    raw_pointing = TelescopeStructurePointingContainer(
        azimuth=0.3 * u.rad,
        altitude=0.7 * u.rad,
    )
    displacement = TelescopeStructureDisplacementContainer(
        delta_azimuth=0.2 * u.rad,
        delta_altitude=-0.1 * u.rad,
    )

    event.monitoring.tel[tel_id].structure_pointing = raw_pointing
    event.monitoring.tel[tel_id].structure_displacement = displacement

    pointing_calibrator(event)

    calibrated = event.monitoring.tel[tel_id].pointing
    assert calibrated is not None
    assert u.isclose(calibrated.azimuth, 0.5 * u.rad)
    assert u.isclose(calibrated.altitude, 0.6 * u.rad)


def test_apply_structure_displacement_wraps_azimuth(pointing_calibrator):
    event = ArrayEventContainer()
    tel_id = 2
    event.trigger.tels_with_trigger = [tel_id]

    raw_pointing = TelescopeStructurePointingContainer(
        azimuth=(2 * 3.141592653589793 - 0.2) * u.rad,
        altitude=1.0 * u.rad,
    )
    displacement = TelescopeStructureDisplacementContainer(
        delta_azimuth=0.3 * u.rad,
        delta_altitude=0.0 * u.rad,
    )

    event.monitoring.tel[tel_id].structure_pointing = raw_pointing
    event.monitoring.tel[tel_id].structure_displacement = displacement

    pointing_calibrator(event)

    calibrated = event.monitoring.tel[tel_id].pointing
    assert calibrated is not None
    assert u.isclose(calibrated.azimuth, 0.1 * u.rad)
    assert u.isclose(calibrated.altitude, 1.0 * u.rad)


def test_missing_structure_pointing_logs_warning(pointing_calibrator, caplog):
    event = ArrayEventContainer()
    tel_id = 3
    event.trigger.tels_with_trigger = [tel_id]

    event.monitoring.tel[
        tel_id
    ].structure_displacement = TelescopeStructureDisplacementContainer(
        delta_azimuth=0.1 * u.rad,
        delta_altitude=0.0 * u.rad,
    )

    pointing_calibrator(event)

    assert event.monitoring.tel[tel_id].pointing is None
    assert "No structure pointing data available." in caplog.text


def test_missing_structure_displacement_logs_warning(pointing_calibrator, caplog):
    event = ArrayEventContainer()
    tel_id = 4
    event.trigger.tels_with_trigger = [tel_id]

    event.monitoring.tel[
        tel_id
    ].structure_pointing = TelescopeStructurePointingContainer(
        azimuth=0.1 * u.rad,
        altitude=0.2 * u.rad,
    )

    pointing_calibrator(event)

    assert event.monitoring.tel[tel_id].pointing is None
    assert "No structure displacement data available." in caplog.text


def test_missing_tel_in_monitoring_is_skipped(pointing_calibrator):
    event = ArrayEventContainer()
    tel_id = 5
    event.trigger.tels_with_trigger = [tel_id]

    pointing_calibrator(event)

    assert tel_id not in event.monitoring.tel


def test_only_matching_telescopes_are_calibrated(pointing_calibrator):
    event = ArrayEventContainer()
    tels = [1, 2]
    event.trigger.tels_with_trigger = tels

    for tel_id in tels:
        event.monitoring.tel[
            tel_id
        ].structure_pointing = TelescopeStructurePointingContainer(
            azimuth=0.1 * u.rad,
            altitude=0.2 * u.rad,
        )
        event.monitoring.tel[
            tel_id
        ].structure_displacement = TelescopeStructureDisplacementContainer(
            delta_azimuth=0.0 * u.rad,
            delta_altitude=0.0 * u.rad,
        )

    event.monitoring.tel[2].structure_pointing = None

    pointing_calibrator(event)

    assert event.monitoring.tel[1].pointing is not None
    assert event.monitoring.tel[2].pointing is None
