""" Tests for CameraGeometry """
import numpy as np
from astropy import units as u
from ctapipe.instrument import CameraDescription, CameraReadout
import pytest

cam_ids = CameraDescription.get_known_camera_names()


def test_construct():
    """ Check we can make a CameraReadout from scratch """
    cam_id = 0
    sampling_rate = u.Quantity(2, u.GHz)
    reference_pulse_shape = np.ones((2, 20)).astype(np.float)
    reference_pulse_sample_width = u.Quantity(0.5, u.ns)
    readout = CameraReadout(
        cam_id=cam_id,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=reference_pulse_sample_width
    )

    assert readout.cam_id == cam_id
    assert readout.sampling_rate == sampling_rate
    assert (readout.reference_pulse_shape == reference_pulse_shape).all()
    assert readout.reference_pulse_sample_width == reference_pulse_sample_width


@pytest.fixture(scope='module')
def readout():
    cam_id = 0
    sampling_rate = u.Quantity(2, u.GHz)
    reference_pulse_shape = np.ones((2, 20)).astype(np.float)
    reference_pulse_sample_width = u.Quantity(0.5, u.ns)
    return CameraReadout(
        cam_id=cam_id,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=reference_pulse_sample_width
    )


def test_reference_pulse_sample_time(readout):
    sample_time = u.Quantity(np.arange(0, 10, 0.5), u.ns)
    assert (readout.reference_pulse_sample_time == sample_time).all()


def test_to_and_from_table(readout):
    """ Check converting to and from an astropy Table """
    tab = readout.to_table()
    readout2 = readout.from_table(tab)

    assert readout.cam_id == readout2.cam_id
    assert readout.sampling_rate == readout2.sampling_rate
    assert np.array_equal(readout.reference_pulse_shape, readout2.reference_pulse_shape)
    assert readout.reference_pulse_sample_width == readout2.reference_pulse_sample_width


def test_write_read(tmpdir, readout):
    """ Check that serialization to disk doesn't lose info """
    filename = str(tmpdir.join('testcamera.fits.gz'))

    readout.to_table().write(filename, overwrite=True)
    readout2 = readout.from_table(filename)

    assert readout.cam_id == readout2.cam_id
    assert readout.sampling_rate == readout2.sampling_rate
    assert np.array_equal(readout.reference_pulse_shape, readout2.reference_pulse_shape)
    assert readout.reference_pulse_sample_width == readout2.reference_pulse_sample_width


def test_equals():
    """ check we can use the == operator """
    cam_id = 0
    sampling_rate = u.Quantity(2, u.GHz)
    reference_pulse_shape = np.ones((2, 20)).astype(np.float)
    reference_pulse_sample_width = u.Quantity(0.5, u.ns)
    readout1 = CameraReadout(
        cam_id=cam_id,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=reference_pulse_sample_width
    )

    readout2 = CameraReadout(
        cam_id=cam_id,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=reference_pulse_sample_width
    )

    readout3 = CameraReadout(
        cam_id=4,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=reference_pulse_sample_width
    )

    readout4 = CameraReadout(
        cam_id=cam_id,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=u.Quantity(1, u.ns)
    )

    assert readout1 is not readout2
    assert readout2 is not readout3
    assert readout3 is not readout4

    assert readout1 == readout2
    assert readout1 != readout3
    assert readout1 != readout4


def test_hashing():
    """" check that hashes are correctly computed """
    cam_id = 0
    sampling_rate = u.Quantity(2, u.GHz)
    reference_pulse_shape = np.ones((2, 20)).astype(np.float)
    reference_pulse_sample_width = u.Quantity(0.5, u.ns)
    readout1 = CameraReadout(
        cam_id=cam_id,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=reference_pulse_sample_width
    )

    readout2 = CameraReadout(
        cam_id=cam_id,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=reference_pulse_sample_width
    )

    readout3 = CameraReadout(
        cam_id=4,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=reference_pulse_sample_width
    )

    assert len({readout1, readout2, readout3}) == 2


@pytest.mark.parametrize("camera_name", cam_ids)
def test_camera_from_name(camera_name):
    """ check we can construct all cameras from name"""
    camera = CameraReadout.from_name(camera_name)
    assert str(camera) == camera_name
