""" Tests for CameraGeometry """
from itertools import combinations

import numpy as np
import pytest
from astropy import units as u

from ctapipe.instrument import CameraReadout, FromNameWarning


def test_construct():
    """Check we can make a CameraReadout from scratch"""
    name = "Unknown"
    sampling_rate = u.Quantity(2, u.GHz)
    reference_pulse_shape = np.ones((2, 20)).astype(np.float64)
    reference_pulse_sample_width = u.Quantity(0.5, u.ns)
    readout = CameraReadout(
        name=name,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=reference_pulse_sample_width,
        n_pixels=1000,
        n_channels=2,
        n_samples=40,
        n_samples_long=80,
    )

    assert readout.name == name
    assert readout.sampling_rate == sampling_rate
    assert (readout.reference_pulse_shape == reference_pulse_shape).all()
    assert readout.reference_pulse_sample_width == reference_pulse_sample_width
    assert readout.n_pixels == 1000
    assert readout.n_channels == 2
    assert readout.n_samples == 40
    assert readout.n_samples_long == 80


@pytest.fixture(scope="module")
def readout():
    name = "Unknown"
    sampling_rate = u.Quantity(2, u.GHz)
    reference_pulse_shape = np.ones((2, 20)).astype(np.float64)
    reference_pulse_sample_width = u.Quantity(0.5, u.ns)
    return CameraReadout(
        name=name,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=reference_pulse_sample_width,
        n_pixels=1000,
        n_channels=2,
        n_samples=40,
        n_samples_long=80,
    )


def test_reference_pulse_sample_time(readout):
    sample_time = u.Quantity(np.arange(0, 10, 0.5), u.ns)
    assert (readout.reference_pulse_sample_time == sample_time).all()


def test_to_and_from_table(readout):
    """Check converting to and from an astropy Table"""
    tab = readout.to_table()
    readout2 = readout.from_table(tab)

    assert readout.name == readout2.name
    assert readout.sampling_rate == readout2.sampling_rate
    assert np.array_equal(readout.reference_pulse_shape, readout2.reference_pulse_shape)
    assert readout.reference_pulse_sample_width == readout2.reference_pulse_sample_width


def test_write_read(tmpdir, readout):
    """Check that serialization to disk doesn't lose info"""
    filename = str(tmpdir.join("testcamera.fits.gz"))

    readout.to_table().write(filename, overwrite=True)
    readout2 = readout.from_table(filename)

    assert readout.name == readout2.name
    assert readout.sampling_rate == readout2.sampling_rate
    assert np.array_equal(readout.reference_pulse_shape, readout2.reference_pulse_shape)
    assert readout.reference_pulse_sample_width == readout2.reference_pulse_sample_width


def test_equals():
    """check we can use the == operator"""
    name = "Unknown"
    sampling_rate = u.Quantity(2, u.GHz)
    reference_pulse_shape = np.ones((2, 20)).astype(np.float64)
    reference_pulse_sample_width = u.Quantity(0.5, u.ns)
    readout1 = CameraReadout(
        name=name,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=reference_pulse_sample_width,
        n_pixels=1000,
        n_channels=2,
        n_samples=40,
        n_samples_long=80,
    )

    readout2 = CameraReadout(
        name=name,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=reference_pulse_sample_width,
        n_pixels=1000,
        n_channels=2,
        n_samples=40,
        n_samples_long=80,
    )

    readout3 = CameraReadout(
        name=4,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=reference_pulse_sample_width,
        n_pixels=1000,
        n_channels=2,
        n_samples=40,
        n_samples_long=80,
    )

    readout4 = CameraReadout(
        name=name,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=u.Quantity(1, u.ns),
        n_pixels=1000,
        n_channels=2,
        n_samples=40,
        n_samples_long=80,
    )

    assert readout1 is not readout2
    assert readout2 is not readout3
    assert readout3 is not readout4

    assert readout1 == readout2
    assert readout1 != readout3
    assert readout1 != readout4


def test_hashing():
    """ " check that hashes are correctly computed"""
    name = "Unknown"
    sampling_rate = u.Quantity(2, u.GHz)
    reference_pulse_shape = np.ones((2, 20)).astype(np.float64)
    reference_pulse_sample_width = u.Quantity(0.5, u.ns)
    readout1 = CameraReadout(
        name=name,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=reference_pulse_sample_width,
        n_pixels=1000,
        n_channels=2,
        n_samples=40,
        n_samples_long=80,
    )

    readout2 = CameraReadout(
        name=name,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=reference_pulse_sample_width,
        n_pixels=1000,
        n_channels=2,
        n_samples=40,
        n_samples_long=80,
    )

    readout3 = CameraReadout(
        name=4,
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=reference_pulse_sample_width,
        n_pixels=1000,
        n_channels=2,
        n_samples=40,
        n_samples_long=80,
    )

    assert len({readout1, readout2, readout3}) == 2


@pytest.mark.parametrize("name", ["LSTCam", "FlashCam", "NectarCam", "CHEC"])
def test_camera_from_name(name, svc_path):
    """check we can construct all cameras from name"""
    with pytest.warns(FromNameWarning):
        camera = CameraReadout.from_name(name)
    assert str(camera) == name


def test_eq(example_subarray):
    """Test telescopes compare equal / non-equal as expected in example subarray"""
    for tel1, tel2 in combinations(example_subarray.tel.values(), 2):
        if tel1.camera_name == tel2.camera_name:
            assert tel1.camera.readout == tel2.camera.readout
        else:
            assert tel1.camera.readout != tel2.camera.readout


def test_hash(example_subarray):
    """Test telescope hashes are equal / non-equal as expected in example subarray"""
    for tel1, tel2 in combinations(example_subarray.tel.values(), 2):
        if tel1.camera_name == tel2.camera_name:
            assert hash(tel1.camera.readout) == hash(tel2.camera.readout)
        else:
            assert hash(tel1.camera.readout) != hash(tel2.camera.readout)


def test_table_roundtrip(prod5_lst):
    """Test we can roundtrip using from_table / to_table"""
    from ctapipe.instrument import CameraReadout

    camera_in = prod5_lst.camera.readout

    table = camera_in.to_table()
    assert table.meta["TAB_VER"] == CameraReadout.CURRENT_TAB_VERSION
    assert table.meta["TAB_TYPE"] == "ctapipe.instrument.CameraReadout"
    camera_out = CameraReadout.from_table(table)

    assert camera_in == camera_out
    assert hash(camera_in) == hash(camera_out)
