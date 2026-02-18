import json
import logging

import pytest
from astropy.io import fits

from ctapipe.core import ToolConfigurationError, run_tool

pytest.importorskip("pyirf")


@pytest.fixture(scope="module")
def dummy_cuts_file(
    gamma_diffuse_full_reco_file,
    proton_full_reco_file,
    event_loader_config_path,
    irf_tmp_path,
):
    from ctapipe.tools.optimize_event_selection import EventSelectionOptimizer

    output_path = irf_tmp_path / "dummy_cuts.fits"
    run_tool(
        EventSelectionOptimizer(),
        argv=[
            f"--gamma-file={gamma_diffuse_full_reco_file}",
            f"--proton-file={proton_full_reco_file}",
            # Use diffuse gammas weighted to electron spectrum as stand-in
            f"--electron-file={gamma_diffuse_full_reco_file}",
            f"--output={output_path}",
            f"--config={event_loader_config_path}",
        ],
    )
    return output_path


@pytest.mark.parametrize("include_background", (False, True))
@pytest.mark.parametrize("spatial_selection_applied", (True, False))
def test_irf_tool(
    gamma_diffuse_full_reco_file,
    proton_full_reco_file,
    event_loader_config_path,
    dummy_cuts_file,
    tmp_path,
    include_background,
    spatial_selection_applied,
):
    from ctapipe.tools.compute_irf import IrfTool

    output_path = tmp_path / "irf.fits.gz"
    output_benchmarks_path = tmp_path / "benchmarks.fits.gz"

    argv = [
        f"--gamma-file={gamma_diffuse_full_reco_file}",
        f"--cuts={dummy_cuts_file}",
        f"--output={output_path}",
        f"--config={event_loader_config_path}",
    ]
    if spatial_selection_applied:
        argv.append("--spatial-selection-applied")

    if include_background:
        argv.append(f"--proton-file={proton_full_reco_file}")
        # Use diffuse gammas weighted to electron spectrum as stand-in
        argv.append(f"--electron-file={gamma_diffuse_full_reco_file}")
    else:
        argv.append("--no-do-background")

    ret = run_tool(IrfTool(), argv=argv)
    assert ret == 0

    assert output_path.exists()
    assert not output_benchmarks_path.exists()
    # Readability by gammapy is tested by pyirf tests, so not repeated here
    with fits.open(output_path) as hdul:
        assert isinstance(hdul["ENERGY DISPERSION"], fits.BinTableHDU)
        assert isinstance(hdul["EFFECTIVE AREA"], fits.BinTableHDU)
        assert isinstance(hdul["PSF"], fits.BinTableHDU)
        if spatial_selection_applied:
            assert isinstance(hdul["RAD_MAX"], fits.BinTableHDU)

        if include_background:
            assert isinstance(hdul["BACKGROUND"], fits.BinTableHDU)

    output_path.unlink()  # Delete output file

    # Include benchmarks
    argv.append(f"--benchmark True")
    ret = run_tool(IrfTool(), argv=argv)
    assert ret == 0
    assert output_path.exists()
    output_path.unlink()  # Delete output file

    # Include benchmarks
    argv.append(f"--benchmark-output={output_benchmarks_path}")
    ret = run_tool(IrfTool(), argv=argv)
    assert ret == 0

    assert output_path.exists()
    assert output_benchmarks_path.exists()
    with fits.open(output_benchmarks_path) as hdul:
        assert isinstance(hdul["ENERGY BIAS RESOLUTION"], fits.BinTableHDU)
        assert isinstance(hdul["ANGULAR RESOLUTION"], fits.BinTableHDU)
        if include_background:
            assert isinstance(hdul["SENSITIVITY"], fits.BinTableHDU)


def test_irf_tool_no_electrons(
    gamma_diffuse_full_reco_file,
    proton_full_reco_file,
    event_loader_config_path,
    dummy_cuts_file,
    tmp_path,
):
    from ctapipe.tools.compute_irf import IrfTool

    output_path = tmp_path / "irf.fits.gz"
    output_benchmarks_path = tmp_path / "benchmarks.fits.gz"
    logpath = tmp_path / "test_irf_tool_no_electrons.log"
    logger = logging.getLogger("ctapipe.tools.compute_irf")
    logger.addHandler(logging.FileHandler(logpath))

    ret = run_tool(
        IrfTool(),
        argv=[
            f"--gamma-file={gamma_diffuse_full_reco_file}",
            f"--proton-file={proton_full_reco_file}",
            f"--cuts={dummy_cuts_file}",
            f"--output={output_path}",
            f"--benchmark-output={output_benchmarks_path}",
            f"--config={event_loader_config_path}",
            f"--log-file={logpath}",
        ],
    )
    assert ret == 0
    assert output_path.exists()
    assert output_benchmarks_path.exists()
    assert "Estimating background without electron file." in logpath.read_text()


def test_irf_tool_only_gammas(
    gamma_diffuse_full_reco_file, event_loader_config_path, dummy_cuts_file, tmp_path
):
    from ctapipe.tools.compute_irf import IrfTool

    output_path = tmp_path / "irf.fits.gz"
    output_benchmarks_path = tmp_path / "benchmarks.fits.gz"

    with pytest.raises(
        ValueError,
        match="At least a proton file required when specifying `do_background`.",
    ):
        run_tool(
            IrfTool(),
            argv=[
                f"--gamma-file={gamma_diffuse_full_reco_file}",
                f"--cuts={dummy_cuts_file}",
                f"--output={output_path}",
                f"--benchmark-output={output_benchmarks_path}",
                f"--config={event_loader_config_path}",
            ],
            raises=True,
        )

    ret = run_tool(
        IrfTool(),
        argv=[
            f"--gamma-file={gamma_diffuse_full_reco_file}",
            f"--cuts={dummy_cuts_file}",
            f"--output={output_path}",
            f"--benchmark-output={output_benchmarks_path}",
            f"--config={event_loader_config_path}",
            "--no-do-background",
        ],
    )
    assert ret == 0
    assert output_path.exists()
    assert output_benchmarks_path.exists()


# TODO: Add test using point-like gammas


def test_point_like_irf_no_theta_cut(
    gamma_diffuse_full_reco_file,
    proton_full_reco_file,
    event_loader_config_path,
    dummy_cuts_file,
    tmp_path,
):
    from ctapipe.irf import OptimizationResult
    from ctapipe.tools.compute_irf import IrfTool

    gh_cuts_path = tmp_path / "gh_cuts.fits"
    cuts = OptimizationResult.read(dummy_cuts_file)
    cuts.spatial_selection_table = None
    cuts.write(gh_cuts_path)
    assert gh_cuts_path.exists()

    output_path = tmp_path / "irf.fits.gz"
    output_benchmarks_path = tmp_path / "benchmarks.fits.gz"

    with pytest.raises(
        ToolConfigurationError,
        match=rf"{gh_cuts_path} does not contain any direction cut",
    ):
        run_tool(
            IrfTool(),
            argv=[
                f"--gamma-file={gamma_diffuse_full_reco_file}",
                f"--proton-file={proton_full_reco_file}",
                # Use diffuse gammas weighted to electron spectrum as stand-in
                f"--electron-file={gamma_diffuse_full_reco_file}",
                f"--cuts={gh_cuts_path}",
                f"--output={output_path}",
                f"--benchmark-output={output_benchmarks_path}",
                f"--config={event_loader_config_path}",
                "--spatial-selection-applied",
            ],
            raises=True,
        )


def test_irf_tool_wrong_cuts(
    gamma_diffuse_full_reco_file, proton_full_reco_file, dummy_cuts_file, tmp_path
):
    from ctapipe.tools.compute_irf import IrfTool

    output_path = tmp_path / "irf.fits.gz"
    output_benchmarks_path = tmp_path / "benchmarks.fits.gz"

    with pytest.raises(RuntimeError):
        run_tool(
            IrfTool(),
            argv=[
                f"--gamma-file={gamma_diffuse_full_reco_file}",
                f"--proton-file={proton_full_reco_file}",
                # Use diffuse gammas weighted to electron spectrum as stand-in
                f"--electron-file={gamma_diffuse_full_reco_file}",
                f"--cuts={dummy_cuts_file}",
                f"--output={output_path}",
                f"--benchmark-output={output_benchmarks_path}",
            ],
            raises=True,
        )

    config_path = tmp_path / "config.json"
    with config_path.open("w") as f:
        json.dump(
            {
                "DL2EventPreprocessor": {
                    "energy_reconstructor": "ExtraTreesRegressor",
                    "geometry_reconstructor": "HillasReconstructor",
                    "gammaness_classifier": "ExtraTreesClassifier",
                    "DL2EventQualityQuery": {
                        "quality_criteria": [
                            # No criteria for minimum event multiplicity
                            ("valid classifier", "ExtraTreesClassifier_is_valid"),
                            ("valid geom reco", "HillasReconstructor_is_valid"),
                            ("valid energy reco", "ExtraTreesRegressor_is_valid"),
                        ],
                    },
                }
            },
            f,
        )

    logpath = tmp_path / "test_irf_tool_wrong_cuts.log"
    logger = logging.getLogger("ctapipe.tools.compute_irf")
    logger.addHandler(logging.FileHandler(logpath))

    ret = run_tool(
        IrfTool(),
        argv=[
            f"--gamma-file={gamma_diffuse_full_reco_file}",
            f"--proton-file={proton_full_reco_file}",
            # Use diffuse gammas weighted to electron spectrum as stand-in
            f"--electron-file={gamma_diffuse_full_reco_file}",
            f"--cuts={dummy_cuts_file}",
            f"--output={output_path}",
            f"--benchmark-output={output_benchmarks_path}",
            f"--config={config_path}",
            f"--log-file={logpath}",
        ],
    )
    assert ret == 0
    assert output_path.exists()
    assert output_benchmarks_path.exists()
    assert (
        "Quality criteria are different from quality criteria "
        "used for calculating g/h / theta cuts." in logpath.read_text()
    )
