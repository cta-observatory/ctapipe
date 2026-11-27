from traitlets.config.loader import Config

from ctapipe.containers import EventType
from ctapipe.core import run_tool
from ctapipe.io import read_table
from ctapipe.tools.calculate_pixel_stats import PixelStatisticsCalculatorTool
from ctapipe.tools.estimate_nsb import EstimateNSBlevelTool
from ctapipe.tools.process import ProcessorTool
from ctapipe.utils import get_dataset_path


def process_dl1_sim_pedestal(ped_file, tmp_path):

    dl1_path = tmp_path / "pedestal_dl1.h5"
    config = Config(
        {
            "EventSource": {
                "skip_calibration_events": False,
            },
            "EventTypeFilter": {
                "allowed_types": {EventType.SKY_PEDESTAL, EventType.PEDESTAL}
            },
            "CameraCalibrator": {
                "image_extractor_type": [("type", "*", "FixedWindowSum")],
            },
            "FixedWindowSum": {
                "window_width": 12,
                "window_shift": 6,
                "apply_integration_correction": False,
            },
        }
    )

    ret = run_tool(
        ProcessorTool(config=config),
        argv=[
            f"--input={ped_file}",
            f"--output={dl1_path}",
            "--write-images",
            "--no-write-parameters",
            "--overwrite",
        ],
    )

    assert ret == 0

    return dl1_path


def calculate_pixel_stat(dl1_path, tmp_path, chunk_size):

    stats_path = tmp_path / "pedestal_pixel_stats.h5"
    config = Config(
        {
            "PixelStatisticsCalculatorTool": {
                "input_column_name": "image",
            },
            "PixelStatisticsCalculator": {
                "stats_aggregator_type": [
                    ("type", "*", "PlainAggregator"),
                ],
            },
            "SizeChunking": {
                "chunk_size": chunk_size,
            },
        }
    )

    ret = run_tool(
        PixelStatisticsCalculatorTool(config=config),
        argv=[f"--input_url={dl1_path}", f"--output_path={stats_path}", "--overwrite"],
    )

    assert ret == 0

    return stats_path


def test_estimate_nsb_tool_simulation(tmp_path):
    """Test EstimateNSBlevelTool execution on simulation data."""
    output_path = tmp_path / "nsb_params.h5"

    nsb_file = get_dataset_path("only_nsb_LaPalma_Alpha__nsb_x0.25.simtel.zst")

    nsb_lvl_sim = 1
    # nsb_file only contains 10 events -> we cannot test more than 2 times wrt the reference NSB (sim_file)
    test_nsb_lvl = [1, 2]

    # Only LSTN-1 is available in simulations
    expected_nsb_level = [
        {
            1: 0
        },  # input and reference pedestal contain the same NSB level -> no additional NSB
        {
            1: 4
        },  # for NSB-only with 0.25*(dark NSB), input (2*dark NSB) and sim (dark NSB) -> add 4 NSB-only waveforms at 0.25 dark NSB
    ]
    tel_ids = [1]

    chunk_size = 100  # same number of events as ped_file
    # check outputs for an input with a single chunk and variance method
    config = Config(
        {
            "EstimateNSBlevelTool": {
                "quantile_threshold": 0.9,
                "min_npixels_valid": 1000,
                "min_nped_events": chunk_size,  # same value as chunk_size in PixelStatisticsCalculator
                "image_extractor_type": "FixedWindowSum",
            },
            "FixedWindowSum": {
                "window_width": 12,
                "window_shift": 6,
                "apply_integration_correction": False,
            },
        }
    )

    for ii, insb_lvl in enumerate(test_nsb_lvl):
        ped_file = get_dataset_path(
            f"nsb-padding/pedestals_run00000{insb_lvl}_za00deg_azm000deg_North_LSTN-01_7.0.1_nsb-{insb_lvl}x.simtel.zst"
        )

        ped_dl1 = process_dl1_sim_pedestal(ped_file, tmp_path)

        input = calculate_pixel_stat(ped_dl1, tmp_path, chunk_size)

        sim_file = get_dataset_path(
            f"nsb-padding/pedestals_run00000{nsb_lvl_sim}_za00deg_azm000deg_North_LSTN-01_7.0.1_nsb-{nsb_lvl_sim}x.simtel.zst"
        )

        ret = run_tool(
            EstimateNSBlevelTool(config=config),
            argv=[
                f"--input-file={input}",
                f"--nsb-file={nsb_file}",
                f"--sim-file={sim_file}",
                f"--output={output_path}",
                "--overwrite",
            ],
        )

        assert ret == 0
        assert output_path.exists()

        for tel_id in tel_ids:
            nsb_tab = read_table(
                tmp_path / "nsb_params.h5",
                path=f"/dl1/monitoring/nsb/tel_{tel_id:03d}",
            )

            assert "exact_nsb_level" in nsb_tab.colnames
            assert "nsb_level" in nsb_tab.colnames
            assert "is_valid" in nsb_tab.colnames

            # The test files contain 100 events and the chunk size is also 100
            assert len(nsb_tab) == 1
            assert nsb_tab["is_valid"].all()
            assert (nsb_tab["nsb_level"] == expected_nsb_level[ii][tel_id]).all()
