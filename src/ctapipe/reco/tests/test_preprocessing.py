import astropy.units as u
import numpy as np
from astropy.table import Table
from numpy.testing import assert_array_equal


def test_table_to_float32():
    from ctapipe.reco.preprocessing import table_to_float

    t = Table({"a": [1.0, 1e50, np.inf, -np.inf, np.nan], "b": [1, 2, 3, 4, 5]})

    fmax = np.finfo(np.float32).max
    fmin = np.finfo(np.float32).min
    expected = np.array(
        [[1.0, fmax, fmax, fmin, np.nan], [1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32
    ).T

    array = table_to_float(t)
    assert array.dtype == np.float32
    assert_array_equal(array, expected)


def test_table_to_float32_units():
    from ctapipe.reco.preprocessing import table_to_float

    t = Table(
        {"a": [1.0, 1e50, np.inf, -np.inf, np.nan] * u.m, "b": [1, 2, 3, 4, 5] * u.deg}
    )

    fmax = np.finfo(np.float32).max
    fmin = np.finfo(np.float32).min
    expected = np.array(
        [[1.0, fmax, fmax, fmin, np.nan], [1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32
    ).T

    array = table_to_float(t)
    assert array.dtype == np.float32
    assert_array_equal(array, expected)


def test_table_to_float64():
    from ctapipe.reco.preprocessing import table_to_float

    t = Table({"a": [1.0, 1e50, np.inf, -np.inf, np.nan], "b": [1, 2, 3, 4, 5]})

    fmax = np.finfo(np.float64).max
    fmin = np.finfo(np.float64).min
    expected = np.array(
        [[1.0, 1e50, fmax, fmin, np.nan], [1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64
    ).T

    array = table_to_float(t, dtype=np.float64)
    assert_array_equal(array, expected)
    assert array.dtype == np.float64
    assert_array_equal(array, expected)


def test_check_valid_rows():
    from ctapipe.reco.preprocessing import check_valid_rows

    t = Table({"a": [1.0, 2, np.inf, np.nan, np.nan], "b": [1, np.inf, 3, 4, 5]})

    valid = check_valid_rows(t, warn=False)
    assert_array_equal(valid, [True, True, True, False, False])


def test_collect_features(example_event, example_subarray):
    from ctapipe.calib import CameraCalibrator
    from ctapipe.image import ImageProcessor
    from ctapipe.reco import ShowerProcessor
    from ctapipe.reco.preprocessing import collect_features

    event = example_event
    subarray = example_subarray

    calib = CameraCalibrator(subarray)
    image_processor = ImageProcessor(subarray)
    shower_processor = ShowerProcessor(subarray)

    calib(event)
    image_processor(event)
    shower_processor(event)

    tel_id = next(iter(event.dl2.tel))
    tab = collect_features(event, tel_id=tel_id)

    k = "HillasReconstructor"
    impact = event.dl2.tel[tel_id].impact[k]
    assert tab[f"{k}_tel_impact_distance"].quantity[0] == impact.distance

    geometry = event.dl2.stereo.geometry[k]
    assert tab[f"{k}_az"].quantity[0] == geometry.az

    hillas = event.dl1.tel[tel_id].parameters.hillas
    assert tab["hillas_intensity"].quantity[0] == hillas.intensity

    leakage = event.dl1.tel[tel_id].parameters.leakage
    assert tab["leakage_intensity_width_1"].quantity[0] == leakage.intensity_width_1

    tab = collect_features(
        event, tel_id=tel_id, subarray_table=subarray.to_table("joined")
    )
    focal_length = subarray.tel[tel_id].optics.equivalent_focal_length
    assert tab["equivalent_focal_length"].quantity[0] == focal_length
