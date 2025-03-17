#!/usr/bin/env python3
"""
Test ctapipe-process on a few different use cases
"""

import json
from subprocess import CalledProcessError

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
import tables
from numpy.testing import assert_allclose, assert_array_equal

from ctapipe.core import run_tool
from ctapipe.instrument.subarray import SubarrayDescription
from ctapipe.io import EventSource, TableLoader, read_table
from ctapipe.io.tests.test_event_source import DummyEventSource
from ctapipe.tools.process import ProcessorTool
from ctapipe.tools.quickstart import CONFIGS_TO_WRITE, QuickStartTool
from ctapipe.utils import get_dataset_path, resource_file

GAMMA_TEST_LARGE = get_dataset_path("gamma_test_large.simtel.gz")
LST_MUONS = get_dataset_path("lst_muons.simtel.zst")


@pytest.mark.parametrize(
    "config_files",
    [
        ("base_config.yaml", "stage1_config.yaml"),
        ("stage1_config.toml",),
        ("stage1_config.json",),
    ],
)
def test_read_yaml_toml_json_config(dl1_image_file, config_files):
    """check that we can read multiple formats of config file"""
    tool = ProcessorTool()

    for config_base in config_files:
        config = resource_file(config_base)
        tool.load_config_file(config)

    tool.config.EventSource.input_url = dl1_image_file
    tool.config.DataWriter.overwrite = True
    tool.setup()
    assert (
        tool.get_current_config()["ProcessorTool"]["DataWriter"]["contact_info"].name
        == "YOUR-NAME-HERE"
    )


def test_multiple_configs(dl1_image_file):
    """ensure a config file loaded later overwrites keys from an earlier one"""
    tool = ProcessorTool()

    tool.load_config_file(resource_file("base_config.yaml"))
    tool.load_config_file(resource_file("stage2_config.yaml"))

    tool.config.EventSource.input_url = dl1_image_file
    tool.config.DataWriter.overwrite = True
    tool.setup()

    # ensure the overwriting works (base config has this option disabled)
    assert tool.get_current_config()["ProcessorTool"]["DataWriter"]["write_dl2"] is True


def test_stage_1_dl1(tmp_path, dl1_image_file, dl1_parameters_file):
    """check simtel to DL1 conversion"""
    config = resource_file("stage1_config.json")

    # DL1A file as input
    dl1b_from_dl1a_file = tmp_path / "dl1b_fromdl1a.dl1.h5"
    run_tool(
        ProcessorTool(),
        argv=[
            f"--config={config}",
            f"--input={dl1_image_file}",
            f"--output={dl1b_from_dl1a_file}",
            "--camera-frame",
            "--write-parameters",
            "--overwrite",
        ],
        cwd=tmp_path,
        raises=True,
    )

    # check tables were written
    with tables.open_file(dl1b_from_dl1a_file, mode="r") as testfile:
        assert testfile.root.dl1
        assert testfile.root.dl1.event.telescope
        assert testfile.root.dl1.event.subarray
        assert testfile.root.configuration.instrument.subarray.layout
        assert testfile.root.configuration.instrument.telescope.optics
        assert testfile.root.configuration.instrument.telescope.camera.geometry_0
        assert testfile.root.configuration.instrument.telescope.camera.readout_0

    # check we can read telescope parameters
    dl1_features = pd.read_hdf(
        dl1b_from_dl1a_file, "/dl1/event/telescope/parameters/tel_025"
    )
    features = (
        "obs_id",
        "event_id",
        "tel_id",
        "camera_frame_hillas_intensity",
        "camera_frame_hillas_x",
        "concentration_cog",
        "leakage_pixels_width_1",
    )
    for feature in features:
        assert feature in dl1_features.columns

    true_impact = read_table(
        dl1b_from_dl1a_file,
        "/simulation/event/telescope/impact/tel_025",
    )
    assert "true_impact_distance" in true_impact.colnames

    # DL1B file as input
    ret = run_tool(
        ProcessorTool(),
        argv=[
            f"--config={config}",
            f"--input={dl1_parameters_file}",
            f"--output={tmp_path}/dl1b_from_dl1b.dl1.h5",
            "--write-parameters",
            "--overwrite",
        ],
        cwd=tmp_path,
        raises=False,
    )
    assert ret == 1


def test_stage1_datalevels(tmp_path):
    """test the dl1 tool on a file not providing r1, dl0 or dl1a"""

    dummy_file = tmp_path / "datalevels_dummy.h5"
    out_file = tmp_path / "datalevels_dummy_stage1_output.h5"
    with dummy_file.open("wb") as infile:
        infile.write(b"dummy")

    config = resource_file("stage1_config.json")
    tool = ProcessorTool()

    with pytest.raises(CalledProcessError):
        run_tool(
            tool,
            argv=[
                f"--config={config}",
                f"--input={dummy_file}",
                f"--output={out_file}",
                "--write-images",
                "--overwrite",
            ],
            cwd=tmp_path,
        )

    # make sure the dummy event source was really used
    assert isinstance(tool.event_source, DummyEventSource)


def test_stage_2_from_simtel(tmp_path, provenance):
    """check we can go to DL2 geometry from simtel file"""
    config = resource_file("stage2_config.json")
    output = tmp_path / "test_stage2_from_simtel.DL2.h5"

    provenance_log = tmp_path / "provenance.log"
    input_path = get_dataset_path("gamma_prod5.simtel.zst")
    run_tool(
        ProcessorTool(),
        argv=[
            f"--config={config}",
            f"--input={input_path}",
            f"--output={output}",
            f"--provenance-log={provenance_log}",
            "--overwrite",
        ],
        cwd=tmp_path,
        raises=True,
    )

    # check tables were written
    with tables.open_file(output, mode="r") as testfile:
        dl2 = read_table(
            testfile,
            "/dl2/event/subarray/geometry/HillasReconstructor",
        )
        subarray = SubarrayDescription.from_hdf(testfile)

        # test tel_ids are included and transformed correctly
        assert "HillasReconstructor_telescopes" in dl2.colnames
        assert dl2["HillasReconstructor_telescopes"].dtype == np.bool_
        assert dl2["HillasReconstructor_telescopes"].shape[1] == len(subarray)

    activities = json.loads(provenance_log.read_text())
    assert len(activities) == 1

    activity = activities[0]
    assert activity["status"] == "completed"
    assert len(activity["input"]) == 2
    assert activity["input"][0]["url"] == str(config)
    assert activity["input"][1]["url"] == str(input_path)

    assert len(activity["output"]) == 1
    assert activity["output"][0]["url"] == str(output)


def test_stage_2_from_dl1_images(tmp_path, dl1_image_file):
    """check we can go to DL2 geometry from DL1 images"""
    config = resource_file("stage2_config.json")
    output = tmp_path / "test_stage2_from_dl1image.DL2.h5"

    run_tool(
        ProcessorTool(),
        argv=[
            f"--config={config}",
            f"--input={dl1_image_file}",
            f"--output={output}",
            "--overwrite",
        ],
        cwd=tmp_path,
        raises=True,
    )

    # check tables were written
    with tables.open_file(output, mode="r") as testfile:
        assert testfile.root.dl2.event.subarray.geometry.HillasReconstructor


def test_stage_2_from_dl1_params(tmp_path, dl1_parameters_file):
    """check we can go to DL2 geometry from DL1 parameters"""

    config = resource_file("stage2_config.json")
    output = tmp_path / "test_stage2_from_dl1param.DL2.h5"

    run_tool(
        ProcessorTool(),
        argv=[
            f"--config={config}",
            f"--input={dl1_parameters_file}",
            f"--output={output}",
            "--overwrite",
        ],
        cwd=tmp_path,
        raises=True,
    )

    # check tables were written
    with tables.open_file(output, mode="r") as testfile:
        assert testfile.root.dl2.event.subarray.geometry.HillasReconstructor


def test_ml_preprocessing_from_simtel(tmp_path):
    """check we can write both dl1 and dl2 info (e.g. for ml_preprocessing input)"""

    config = resource_file("ml_preprocessing_config.json")
    output = tmp_path / "test_ml_preprocessing.DL1DL2.h5"

    run_tool(
        ProcessorTool(),
        argv=[
            f"--config={config}",
            f"--input={GAMMA_TEST_LARGE}",
            f"--output={output}",
            "--max-events=5",
            "--overwrite",
            "--SimTelEventSource.focal_length_choice=EQUIVALENT",
        ],
        cwd=tmp_path,
        raises=True,
    )

    # check tables were written
    with tables.open_file(output, mode="r") as testfile:
        assert testfile.root.dl1.event.telescope.parameters.tel_002
        assert testfile.root.dl2.event.subarray.geometry.HillasReconstructor


def test_image_modifications(tmp_path, dl1_image_file):
    """
    Test that running ctapipe-process with an ImageModifier set
    produces a file with different images.
    """

    unmodified_images = read_table(
        dl1_image_file, "/dl1/event/telescope/images/tel_025"
    )
    noise_config = resource_file("image_modification_config.json")

    dl1_modified = tmp_path / "dl1_modified.dl1.h5"
    run_tool(
        ProcessorTool(),
        argv=[
            f"--config={noise_config}",
            f"--input={dl1_image_file}",
            f"--output={dl1_modified}",
            "--write-parameters",
            "--overwrite",
        ],
        cwd=tmp_path,
        raises=True,
    )
    modified_images = read_table(dl1_modified, "/dl1/event/telescope/images/tel_025")
    # Test that significantly more light is recorded (bias in dim pixels)
    assert modified_images["image"].sum() / unmodified_images["image"].sum() > 1.5


@pytest.mark.parametrize(
    "filename", ["base_config.yaml", "stage1_config.json", "stage1_config.toml"]
)
def test_quickstart_templates(filename):
    """ensure template configs have an appropriate placeholder for the contact info"""
    config = resource_file(filename)
    text = config.read_text()

    assert "YOUR-NAME-HERE" in text, "Missing expected name placeholder"
    assert "YOUREMAIL@EXAMPLE.ORG" in text, "Missing expected email placeholder"
    assert "YOUR-ORGANIZATION" in text, "Missing expected org placeholder"


def test_quickstart(tmp_path):
    """ensure quickstart tool generates expected output"""

    tool = QuickStartTool()
    run_tool(
        tool,
        cwd=tmp_path,
        argv=[
            "--workdir",
            "ProdX",
            "--name",
            "test",
            "--email",
            "a@b.com",
            "--org",
            "CTA",
        ],
    )

    assert (tmp_path / "ProdX" / "README.md").exists()

    for config in CONFIGS_TO_WRITE:
        assert (tmp_path / "ProdX" / config).exists()


def test_read_from_simtel_and_dl1(prod5_proton_simtel_path, tmp_path):
    """In #2057 reading a subset of allowed tels from
    simtel yields another result as reading from DL1,
    which was produced with all tels.

    This test has three steps:
    1) Create a DL2 file from simtel.
    2) Create a DL1 file from simtel.
    3) Create from that DL1 file another DL2 file.

    Keep in mind that both DL2 allowed_tels need to be the same,
    but different from the allowed_tels in the simtel->DL1 step!
    """

    input_path = prod5_proton_simtel_path

    many_tels = [
        f"--EventSource.allowed_tels={i}"
        for i in (30, 100, 102, 105, 106, 108, 111, 112, 113, 114, 115, 121, 122, 128)
    ]
    few_tels = [
        f"--EventSource.allowed_tels={i}" for i in (102, 108, 111, 112, 121, 122, 128)
    ]

    # 1) Create DL2 from simtel.
    dl2_from_simtel = tmp_path / "from_simtel.dl2.h5"
    argv = [
        f"--input={input_path}",
        f"--output={dl2_from_simtel}",
        "--write-showers",
        "--write-parameters",
        "--progress",
        "--EventSource.focal_length_choice=EQUIVALENT",
    ] + few_tels
    assert run_tool(ProcessorTool(), argv=argv, cwd=tmp_path) == 0

    # 2) Create DL1 from simtel.
    dl1_from_simtel = tmp_path / "from_simtel.dl1.h5"
    argv = [
        f"--input={input_path}",
        f"--output={dl1_from_simtel}",
        "--write-showers",
        "--write-parameters",
        "--progress",
        "--EventSource.focal_length_choice=EQUIVALENT",
    ] + many_tels
    assert run_tool(ProcessorTool(), argv=argv, cwd=tmp_path) == 0

    # 3) Create from that DL1 file another DL2 file.
    dl2_from_dl1 = tmp_path / "from_dl1.dl2.h5"
    argv = [
        f"--input={dl1_from_simtel}",
        f"--output={dl2_from_dl1}",
        "--write-showers",
        "--write-parameters",
        "--progress",
        "--EventSource.focal_length_choice=EQUIVALENT",
    ] + few_tels
    assert run_tool(ProcessorTool(), argv=argv, cwd=tmp_path) == 0

    with TableLoader(dl2_from_simtel) as loader:
        events_from_simtel = loader.read_subarray_events()
    with TableLoader(dl2_from_dl1) as loader:
        events_from_dl1 = loader.read_subarray_events()

    with tables.open_file(dl2_from_dl1) as f:
        assert "/simulation/service/shower_distribution" in f.root

    # both files should contain identical data
    assert_array_equal(events_from_simtel["event_id"], events_from_dl1["event_id"])

    assert_allclose(
        events_from_simtel["HillasReconstructor_core_x"],
        events_from_dl1["HillasReconstructor_core_x"],
    )

    # regression test: before the simulation iterator was not incremented,
    # so the simulated events don't match the reconstructed events
    assert_allclose(
        events_from_simtel["true_core_x"],
        events_from_dl1["true_core_x"],
    )


def test_muon_reconstruction_simtel(tmp_path):
    """ensure processor tool generates expected output when used to analyze muons"""
    pytest.importorskip("iminuit")

    muon_simtel_output_file = tmp_path / "muon_reco_on_simtel.h5"
    run_tool(
        ProcessorTool(),
        argv=[
            f"--input={LST_MUONS}",
            f"--output={muon_simtel_output_file}",
            "--SimTelEventSource.focal_length_choice=EQUIVALENT",
            "--overwrite",
            "--write-muon-parameters",
        ],
        cwd=tmp_path,
        raises=True,
    )

    table = read_table(muon_simtel_output_file, "/dl1/event/telescope/muon/tel_001")
    assert len(table) > 20
    assert np.count_nonzero(np.isfinite(table["muonring_radius"])) > 0
    assert np.all(
        np.logical_or(
            np.isfinite(table["muonring_radius"]),
            np.isnan(table["muonring_radius"]),
        )
    )

    with EventSource(
        muon_simtel_output_file, focal_length_choice="EQUIVALENT"
    ) as source:
        radius = table["muonring_radius"].quantity
        efficiency = table["muonefficiency_optical_efficiency"]
        completeness = table["muonparameters_completeness"]

        for event in source:
            muon = event.muon.tel[1]
            assert u.isclose(muon.ring.radius, radius[event.count], equal_nan=True)
            assert np.isclose(
                muon.parameters.completeness, completeness[event.count], equal_nan=True
            )
            assert np.isclose(
                muon.efficiency.optical_efficiency,
                efficiency[event.count],
                equal_nan=True,
            )


def test_plugin_help(capsys):
    ProcessorTool().print_help(classes=True)
    captured = capsys.readouterr()
    assert (
        "PluginEventSource.foo" in captured.out
    ), "Tool help is missing plugin classes, did you run `pip install -e ./test_plugin`?"
    assert (
        "PluginReconstructor.foo" in captured.out
    ), "Tool help is missing plugin classes, did you run `pip install -e ./test_plugin`?"


def test_only_trigger_and_simulation(tmp_path):
    output = tmp_path / "only_trigger_and_simulation.h5"

    run_tool(
        ProcessorTool(),
        argv=[
            "--input=dataset://gamma_prod5.simtel.zst",
            f"--output={output}",
            "--no-write-parameters",
            "--overwrite",
        ],
        cwd=tmp_path,
        raises=True,
    )

    with TableLoader(output) as loader:
        events = loader.read_subarray_events(dl2=False)
        assert len(events) == 7
        assert "tels_with_trigger" in events.colnames
        assert "true_energy" in events.colnames


@pytest.mark.parametrize(
    ("input_url", "args"),
    [
        pytest.param(
            "dataset://gamma_diffuse_dl2_train_small.dl2.h5",
            ["--no-write-images", "--max-events=20"],
            id="0.17",
        )
    ],
)
def test_on_old_file(input_url, args, tmp_path):
    config = resource_file("stage1_config.json")

    output_path = tmp_path / "test.dl1.h5"
    run_tool(
        ProcessorTool(),
        argv=[
            f"--config={config}",
            f"--input={input_url}",
            f"--output={output_path}",
            "--write-showers",
            "--overwrite",
            *args,
        ],
        cwd=tmp_path,
        raises=True,
    )

    with tables.open_file(output_path) as f:
        assert "/configuration/telescope/pointing" in f.root

    with TableLoader(output_path) as loader:
        events = loader.read_subarray_events()

        # check that we have valid reconstructions and that in case
        # we don't, is_valid is False, regression test for #2651
        finite_reco = np.isfinite(events["HillasReconstructor_alt"])
        assert np.any(finite_reco)
        np.testing.assert_array_equal(
            finite_reco, events["HillasReconstructor_is_valid"]
        )


def test_prod6_issues(tmp_path):
    """Test behavior of source on file from prod6, see issues #2344 and #2660"""
    input_url = "dataset://prod6_issues.simtel.zst"
    output_path = tmp_path / "test.dl1.h5"

    run_tool(
        ProcessorTool(),
        argv=[
            f"--input={input_url}",
            f"--output={output_path}",
            "--write-images",
            "--write-showers",
            "--overwrite",
        ],
        cwd=tmp_path,
        raises=True,
    )

    with TableLoader(output_path) as loader:
        tel_events = loader.read_telescope_events()
        subarray_events = loader.read_subarray_events()

        trigger_counts = np.count_nonzero(subarray_events["tels_with_trigger"], axis=0)
        _, tel_event_counts = np.unique(tel_events["tel_id"], return_counts=True)

        mask = trigger_counts > 0
        np.testing.assert_equal(trigger_counts[mask], tel_event_counts)

        images = loader.read_telescope_events([32], true_images=True)
        images.add_index("event_id")
        np.testing.assert_array_equal(images.loc[1664106]["true_image"], -1)
