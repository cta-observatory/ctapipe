#!/usr/bin/env python3

import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import tables
from astropy import units as u
from traitlets.config import Config

from ctapipe.calib import CameraCalibrator
from ctapipe.containers import (
    ParticleClassificationContainer,
    ReconstructedEnergyContainer,
    ReconstructedGeometryContainer,
)
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import DataLevel, EventSource
from ctapipe.io.datawriter import DATA_MODEL_VERSION, DataWriter
from ctapipe.io.hdf5tableio import get_column_attrs
from ctapipe.utils import get_dataset_path


def generate_dummy_dl2(event):
    """generate some dummy DL2 info and see if we can write it"""

    algos = ["HillasReconstructor", "ImPACTReconstructor"]

    for algo in algos:
        for tel_id in event.dl1.tel:
            event.dl2.tel[tel_id].geometry[algo] = ReconstructedGeometryContainer(
                alt=70 * u.deg,
                az=120 * u.deg,
                prefix=f"{algo}_tel",
            )

            event.dl2.tel[tel_id].energy[algo] = ReconstructedEnergyContainer(
                energy=10 * u.TeV,
                prefix=f"{algo}_tel",
            )
            event.dl2.tel[tel_id].classification[
                algo
            ] = ParticleClassificationContainer(
                prediction=0.9,
                prefix=f"{algo}_tel",
            )

        event.dl2.stereo.geometry[algo] = ReconstructedGeometryContainer(
            alt=72 * u.deg,
            az=121 * u.deg,
            telescopes=[1, 2, 4],
            prefix=algo,
        )

        event.dl2.stereo.energy[algo] = ReconstructedEnergyContainer(
            energy=10 * u.TeV,
            telescopes=[1, 2, 4],
            prefix=algo,
        )
        event.dl2.stereo.classification[algo] = ParticleClassificationContainer(
            prediction=0.9,
            telescopes=[1, 2, 4],
            prefix=algo,
        )


def test_write(tmpdir: Path):
    """
    Check that we can write and read data from R0-DL2 to files
    """

    output_path = Path(tmpdir / "events.dl1.h5")
    source = EventSource(
        get_dataset_path("gamma_prod5.simtel.zst"),
        focal_length_choice="EQUIVALENT",
    )
    calibrate = CameraCalibrator(subarray=source.subarray)

    with DataWriter(
        event_source=source,
        output_path=output_path,
        write_dl1_parameters=False,
        write_dl1_images=True,
        write_dl2=True,
        write_r0_waveforms=True,
        write_r1_waveforms=True,
    ) as writer:
        writer.log.level = logging.DEBUG
        for event in source:
            calibrate(event)
            generate_dummy_dl2(event)
            writer(event)
        writer.write_simulated_shower_distributions(
            source.simulated_shower_distributions
        )

    assert output_path.exists()

    # check we can get the subarray description:
    sub = SubarrayDescription.from_hdf(output_path)
    assert sub.n_tels > 0

    # check a few things in the output just to make sure there is output. For a
    # full test of the data model, a verify tool should be created.
    with tables.open_file(output_path) as h5file:
        # check R0:
        r0tel = h5file.get_node("/r0/event/telescope/tel_004")
        assert r0tel.col("waveform").max() > 0

        # check R1:
        r1tel = h5file.get_node("/r1/event/telescope/tel_004")
        assert r1tel.col("waveform").max() > 0

        # check DL1:
        images = h5file.get_node("/dl1/event/telescope/images/tel_004")
        assert images.col("image").max() > 0.0
        assert (
            h5file.root._v_attrs["CTA PRODUCT DATA MODEL VERSION"]  # pylint: disable=protected-access
            == DATA_MODEL_VERSION
        )
        shower = h5file.get_node("/simulation/event/subarray/shower")
        shower_attrs = get_column_attrs(shower)
        assert len(shower) > 0
        assert shower.col("true_alt").mean() > 0.0
        assert shower_attrs["true_alt"]["UNIT"] == "deg"

        # check DL2
        for prefix in ("ImPACTReconstructor", "HillasReconstructor"):
            dl2_energy = h5file.get_node(f"/dl2/event/subarray/energy/{prefix}")
            assert np.allclose(dl2_energy.col(f"{prefix}_energy"), 10)
            assert np.count_nonzero(dl2_energy.col(f"{prefix}_telescopes")[0]) == 3

            dl2_tel_energy = h5file.get_node(
                f"/dl2/event/telescope/energy/{prefix}/tel_004"
            )
            assert np.allclose(dl2_tel_energy.col(f"{prefix}_tel_energy"), 10)
            assert "telescopes" not in dl2_tel_energy


def test_roundtrip(tmpdir: Path):
    """
    Check that we can write DL1+DL2 info to files and read them back

    Parameters
    ----------
    tmpdir :
        temp directory fixture
    """

    output_path = Path(tmpdir / "events.DL1DL2.h5")
    source = EventSource(
        get_dataset_path("gamma_prod5.simtel.zst"),
        focal_length_choice="EQUIVALENT",
    )
    calibrate = CameraCalibrator(subarray=source.subarray)

    events = []

    with DataWriter(
        event_source=source,
        output_path=output_path,
        write_dl1_parameters=False,
        write_dl1_images=True,
        transform_image=True,
        image_dtype="int32",
        image_scale=10,
        transform_peak_time=True,
        peak_time_dtype="int16",
        peak_time_scale=100,
        write_dl2=True,
    ) as write:
        write.log.level = logging.DEBUG
        for event in source:
            calibrate(event)
            generate_dummy_dl2(event)
            write(event)
            events.append(deepcopy(event))
        write.write_simulated_shower_distributions(
            source.simulated_shower_distributions
        )
        assert DataLevel.DL1_IMAGES in write.datalevels
        assert DataLevel.DL1_PARAMETERS not in write.datalevels
        assert DataLevel.DL2 in write.datalevels

    assert output_path.exists()

    # check we can get the subarray description:
    sub = SubarrayDescription.from_hdf(output_path)
    assert sub.n_tels > 0

    # check a few things in the output just to make sure there is output. For a
    # full test of the data model, a verify tool should be created.
    with tables.open_file(output_path) as h5file:
        images = h5file.get_node("/dl1/event/telescope/images/tel_004")

        assert len(images) > 0
        assert images.col("image").dtype == np.int32
        assert images.col("peak_time").dtype == np.int16
        assert images.col("image").max() > 0.0

    # make sure it is readable by the event source and matches the images

    with EventSource(output_path) as source:
        for event in source:
            for tel_id, dl1 in event.dl1.tel.items():
                original_image = events[event.count].dl1.tel[tel_id].image
                read_image = dl1.image
                assert np.allclose(original_image, read_image, atol=0.1)

                original_peaktime = events[event.count].dl1.tel[tel_id].peak_time
                read_peaktime = dl1.peak_time
                assert np.allclose(original_peaktime, read_peaktime, atol=0.01)


def test_dl1writer_no_events(tmpdir: Path):
    """
    Check that we can write DL1 files even when no events are given

    Parameters
    ----------
    tmpdir :
        temp directory fixture
    """

    output_path = Path(tmpdir / "no_events.dl1.h5")
    with EventSource("dataset://gamma_prod5.simtel.zst") as source:
        # exhaust source
        for _ in source:
            pass

    assert source.file_.histograms is not None

    with DataWriter(
        event_source=source,
        output_path=output_path,
        write_dl1_parameters=True,
        write_dl1_images=True,
    ) as writer:
        writer.log.level = logging.DEBUG
        writer.write_simulated_shower_distributions(
            source.simulated_shower_distributions
        )

    assert output_path.exists()

    # check we can get the subarray description:
    sub = SubarrayDescription.from_hdf(output_path)
    assert sub == source.subarray

    with tables.open_file(output_path) as h5file:
        assert h5file.get_node("/configuration/simulation/run") is not None
        assert h5file.get_node("/simulation/service/shower_distribution") is not None


def test_metadata(tmpdir: Path):
    output_path = Path(tmpdir / "metadata.dl1.h5")

    dataset = "gamma_20deg_0deg_run2___cta-prod5-paranal_desert-2147m-Paranal-dark_cone10-100evts.simtel.zst"

    config = Config(
        {
            "DataWriter": {
                "Contact": {
                    "name": "Maximilian Nöthe",
                    "email": "maximilian.noethe@tu-dortmund.de",
                    "organization": "TU Dortmund",
                },
                "Instrument": {"site": "CTA-North", "id_": "alpha"},
                "context_metadata": {"EXAMPLE": "test_value"},
            }
        }
    )

    with EventSource(get_dataset_path(dataset)) as source:
        with DataWriter(
            event_source=source,
            output_path=output_path,
            write_dl1_parameters=True,
            write_dl1_images=True,
            config=config,
        ):
            pass

        assert output_path.exists()

        with tables.open_file(output_path) as h5file:
            meta = h5file.root._v_attrs
            assert meta["CTA CONTACT NAME"] == "Maximilian Nöthe"
            assert meta["CTA CONTACT EMAIL"] == "maximilian.noethe@tu-dortmund.de"
            assert meta["CTA CONTACT ORGANIZATION"] == "TU Dortmund"
            assert meta["CTA INSTRUMENT SITE"] == "CTA-North"
            assert meta["CTA INSTRUMENT ID"] == "alpha"
            assert meta["CONTEXT EXAMPLE"] == "test_value"


def test_write_only_r1(r1_hdf5_file):
    with tables.open_file(r1_hdf5_file, "r") as f:
        assert "r1/event/telescope/tel_001" in f.root
