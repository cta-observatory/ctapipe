#!/usr/bin/env python3

import numpy as np
from ctapipe.io.dl1writer import DL1Writer, DL1_DATA_MODEL_VERSION
from ctapipe.utils import get_dataset_path
from ctapipe.io import EventSource
from ctapipe.calib import CameraCalibrator
from pathlib import Path
from ctapipe.instrument import SubarrayDescription
from copy import deepcopy
import tables
import logging


def test_dl1writer(tmpdir: Path):
    """
    Check that we can write DL1 files

    Parameters
    ----------
    tmpdir :
        temp directory fixture
    """

    output_path = Path(tmpdir / "events.dl1.h5")
    source = EventSource(
        get_dataset_path("gamma_LaPalma_baseline_20Zd_180Az_prod3b_test.simtel.gz"),
        max_events=20,
        allowed_tels=[1, 2, 3, 4],
    )
    calibrate = CameraCalibrator(subarray=source.subarray)

    with DL1Writer(
        event_source=source,
        output_path=output_path,
        write_parameters=False,
        write_images=True,
    ) as write_dl1:
        write_dl1.log.level = logging.DEBUG
        for event in source:
            calibrate(event)
            write_dl1(event)
        write_dl1.write_simulation_histograms(source)

    assert output_path.exists()

    # check we can get the subarray description:
    sub = SubarrayDescription.from_hdf(output_path)
    assert sub.num_tels > 0

    # check a few things in the output just to make sure there is output. For a
    # full test of the data model, a verify tool should be created.
    with tables.open_file(output_path) as h5file:
        images = h5file.get_node("/dl1/event/telescope/images/tel_001")
        assert images.col("image").max() > 0.0
        assert (
            h5file.root._v_attrs[
                "CTA PRODUCT DATA MODEL VERSION"
            ]  # pylint: disable=protected-access
            == DL1_DATA_MODEL_VERSION
        )
        shower = h5file.get_node("/simulation/event/subarray/shower")
        assert len(shower) > 0
        assert shower.col("true_alt").mean() > 0.0
        assert (
            shower._v_attrs["true_alt_UNIT"] == "deg"
        )  # pylint: disable=protected-access


def test_dl1writer_int(tmpdir: Path):
    """
    Check that we can write DL1 files

    Parameters
    ----------
    tmpdir :
        temp directory fixture
    """

    output_path = Path(tmpdir / "events.dl1.h5")
    source = EventSource(
        get_dataset_path("gamma_LaPalma_baseline_20Zd_180Az_prod3b_test.simtel.gz"),
        max_events=20,
        allowed_tels=[1, 2, 3, 4],
    )
    calibrate = CameraCalibrator(subarray=source.subarray)

    events = []

    with DL1Writer(
        event_source=source,
        output_path=output_path,
        write_parameters=False,
        write_images=True,
        transform_image=True,
        image_dtype="int32",
        image_scale=10,
        transform_peak_time=True,
        peak_time_dtype="int16",
        peak_time_scale=100,
    ) as write_dl1:
        write_dl1.log.level = logging.DEBUG
        for event in source:
            calibrate(event)
            write_dl1(event)
            events.append(deepcopy(event))
        write_dl1.write_simulation_histograms(source)

    assert output_path.exists()

    # check we can get the subarray description:
    sub = SubarrayDescription.from_hdf(output_path)
    assert sub.num_tels > 0

    # check a few things in the output just to make sure there is output. For a
    # full test of the data model, a verify tool should be created.
    with tables.open_file(output_path) as h5file:
        images = h5file.get_node("/dl1/event/telescope/images/tel_001")

        assert len(images) > 0
        assert images.col("image").dtype == np.int32
        assert images.col("peak_time").dtype == np.int16
        assert images.col("image").max() > 0.0

    # make sure it is readable by the event source and matches the images

    for event in EventSource(output_path):

        for tel_id, dl1 in event.dl1.tel.items():
            original_image = events[event.count].dl1.tel[tel_id].image
            read_image = dl1.image
            assert np.allclose(original_image, read_image, atol=0.1)

            original_peaktime = events[event.count].dl1.tel[tel_id].peak_time
            read_peaktime = dl1.peak_time
            assert np.allclose(original_peaktime, read_peaktime, atol=0.01)
