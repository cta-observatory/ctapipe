from ctapipe.calib import CameraCalibrator
from ctapipe.io import EventSource
from ctapipe.image.time_next_neighbor_cleaning import TimeNextNeighborCleaning
import numpy as np
import astropy.units as u


def test_import():

    cleaning = TimeNextNeighborCleaning()


def test_clean(example_subarray, example_event):

    calib = CameraCalibrator(example_subarray)
    calib(example_event)

    tel_id = 13
    camera = example_subarray.tel[tel_id].camera
    sample_time = 1 / camera.readout.sampling_rate

    image = example_event.dl1.tel[tel_id].image
    image[image <= 0.1] = 0.1
    peaktime = example_event.dl1.tel[tel_id].peak_time

    cleaning = TimeNextNeighborCleaning()
    cleaning.IPR_dict = {
        camera.camera_name: {
            "charge": np.geomspace(0.1, 1000, 100),
            "rate": np.linspace(100, 0, 100),
        }
    }
    mask, boundary = cleaning.clean(
        camera.geometry, image, peaktime, sample_time, sum_time=1 * u.ns
    )
    assert len(mask) == camera.geometry.n_pixels


def test_calculate_ipr(tmp_path):
    source = EventSource("dataset://gamma_test_large.simtel.gz")
    subarray = source.subarray
    calib = CameraCalibrator(subarray=subarray)
    cleaning = TimeNextNeighborCleaning()
    charge_steps = np.linspace(0, 20, 100)
    cleaning.calculate_ipr_from_source(
        source=source, calibrator=calib, charge_steps=charge_steps
    )
    path = tmp_path / "test_ipr.json"
    cleaning.dump_individual_pixel_rates(path)
    cleaning.load_individual_pixel_rates(path)
