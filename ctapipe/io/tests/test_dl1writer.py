#!/usr/bin/env python3

from ctapipe.io.dl1writer import DL1Writer
from ctapipe.utils import get_dataset_path
from ctapipe.io import event_source
from ctapipe.calib import CameraCalibrator
from pathlib import Path
import tables


def test_dl1writer(tmpdir: Path):
    """
    Check that we can write DL1 files

    Parameters
    ----------
    tmpdir :
        temp directory fixture
    """

    output_path = Path(tmpdir / "events.dl1.h5")
    source = event_source(
        get_dataset_path("gamma_test_large.simtel.gz"),
        max_events=20,
        allowed_tels=[1, 2, 3, 4],
    )
    calibrate = CameraCalibrator(subarray=source.subarray)

    with DL1Writer(
        event_source=source, output_path=output_path, write_parameters=False
    ) as write_dl1:
        for event in source:
            calibrate(event)
            write_dl1(event)

    assert output_path.exists()

    with tables.open_file(output_path) as h5file:
        images = h5file.get_node("/dl1/event/telescope/images/tel_001")
        assert len(images.col("image")) > 0
