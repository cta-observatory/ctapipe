import logging
import os

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

    # Do "point-like" cuts to have both g/h and theta cuts in the file
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
            "--point-like",
        ],
    )
    return output_path


@pytest.mark.parametrize("include_bkg", (False, True))
@pytest.mark.parametrize("point_like", (True, False))
def test_irf_tool(
    gamma_diffuse_full_reco_file,
    proton_full_reco_file,
    event_loader_config_path,
    dummy_cuts_file,
    tmp_path,
    include_bkg,
    point_like,
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
    if point_like:
        argv.append("--point-like")

    if include_bkg:
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
        if point_like:
            assert isinstance(hdul["RAD_MAX"], fits.BinTableHDU)
        else:
            assert isinstance(hdul["PSF"], fits.BinTableHDU)

        if include_bkg:
            assert isinstance(hdul["BACKGROUND"], fits.BinTableHDU)

    os.remove(output_path)  # Delete output file

    # Include benchmarks
    argv.append(f"--benchmark-output={output_benchmarks_path}")
    ret = run_tool(IrfTool(), argv=argv)
    assert ret == 0

    assert output_path.exists()
    assert output_benchmarks_path.exists()
    with fits.open(output_benchmarks_path) as hdul:
        assert isinstance(hdul["ENERGY BIAS RESOLUTION"], fits.BinTableHDU)
        assert isinstance(hdul["ANGULAR RESOLUTION"], fits.BinTableHDU)
        if include_bkg:
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
    tmp_path,
):
    from ctapipe.tools.compute_irf import IrfTool
    from ctapipe.tools.optimize_event_selection import EventSelectionOptimizer

    gh_cuts_path = tmp_path / "gh_cuts.fits"
    # Without the "--point-like" flag only G/H cuts are produced.
    run_tool(
        EventSelectionOptimizer(),
        argv=[
            f"--gamma-file={gamma_diffuse_full_reco_file}",
            f"--proton-file={proton_full_reco_file}",
            # Use diffuse gammas weighted to electron spectrum as stand-in
            f"--electron-file={gamma_diffuse_full_reco_file}",
            f"--output={gh_cuts_path}",
            f"--config={event_loader_config_path}",
        ],
    )
    assert gh_cuts_path.exists()

    output_path = tmp_path / "irf.fits.gz"
    output_benchmarks_path = tmp_path / "benchmarks.fits.gz"

    with pytest.raises(
        ToolConfigurationError,
        match=r"Computing a point-like IRF requires an \(optimized\) theta cut.",
    ):
        run_tool(
            IrfTool(),
            argv=[
                f"--gamma-file={gamma_diffuse_full_reco_file}",
                f"--proton-file={proton_full_reco_file}",
                f"--electron-file={gamma_diffuse_full_reco_file}",
                f"--cuts={gh_cuts_path}",
                f"--output={output_path}",
                f"--benchmark-output={output_benchmarks_path}",
                f"--config={event_loader_config_path}",
                "--point-like",
            ],
            raises=True,
        )
