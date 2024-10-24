import json
import os

import pytest
from astropy.io import fits

from ctapipe.core import run_tool


@pytest.fixture(scope="module")
def event_loader_config_path(irf_event_loader_test_config, irf_tmp_path):
    config_path = irf_tmp_path / "event_loader_config.json"
    with config_path.open("w") as f:
        json.dump(irf_event_loader_test_config, f)

    return config_path


@pytest.fixture(scope="module")
def dummy_cuts_file(
    gamma_diffuse_full_reco_file,
    proton_full_reco_file,
    event_loader_config_path,
    irf_tmp_path,
):
    from ctapipe.tools.optimize_event_selection import IrfEventSelector

    # Do "point-like" cuts to have both g/h and theta cuts in the file
    output_path = irf_tmp_path / "dummy_cuts.fits"
    run_tool(
        IrfEventSelector(),
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
    from ctapipe.tools.make_irf import IrfTool

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


# TODO: Add test using point-like gammas
