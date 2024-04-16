def test_collect_features(example_event, example_subarray):
    from ctapipe.calib import CameraCalibrator
    from ctapipe.image import ImageProcessor
    from ctapipe.reco import ShowerProcessor
    from ctapipe.vectorization import collect_features

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
