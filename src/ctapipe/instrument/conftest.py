"""
Common test fixtures for the instrument module
"""

import os
import shutil

import astropy.units as u
import numpy as np
import pytest

from ctapipe.utils.filelock import FileLock


@pytest.fixture(scope="session")
def instrument_dir(tmp_path_factory):
    """Dump instrument of prod5 subarray into a directory as fits"""
    from ctapipe.core import run_tool
    from ctapipe.tools.dump_instrument import DumpInstrumentTool

    path = tmp_path_factory.mktemp("instrument")

    with FileLock(path / ".lock"):
        if (path / "optics.fits.gz").is_file():
            return path

        argv = [
            "--input=dataset://gamma_prod5.simtel.zst",
            f"--outdir={path}",
            "--format=fits",
        ]
        assert run_tool(DumpInstrumentTool(), argv=argv, cwd=path) == 0
        optics_path = path / "MonteCarloArray.optics.fits.gz"
        shutil.move(optics_path, path / "optics.fits.gz")
        return path


@pytest.fixture(scope="function")
def svc_path(instrument_dir):
    before = os.getenv("CTAPIPE_SVC_PATH")
    os.environ["CTAPIPE_SVC_PATH"] = str(instrument_dir.absolute())
    yield
    if before is None:
        del os.environ["CTAPIPE_SVC_PATH"]
    else:
        os.environ["CTAPIPE_SVC_PATH"] = before


@pytest.fixture()
def geometry_hexgrid_square_pixels():
    """A camera with square pixels on a hexagonal grid"""
    from ctapipe.coordinates import CameraFrame
    from ctapipe.instrument import CameraGeometry, PixelGridType, PixelShape

    size = 22
    pix_id = np.arange(256)

    coords = np.arange(-7.5 * size, 7.6 * size, size)
    pix_x, pix_y = np.meshgrid(coords, coords)

    # offset every second row by half the pixel size
    pix_x[::2] += 0.5 * size
    pix_x = pix_x.ravel() * u.mm
    pix_y = pix_y.ravel() * u.mm

    # introduce some gaps
    pix_area = (size / 1.05) ** 2
    pix_area = np.full(len(pix_id), pix_area) * u.mm**2

    geom = CameraGeometry(
        pix_id=pix_id,
        pix_x=pix_x,
        pix_y=pix_y,
        pix_area=pix_area,
        pix_type=PixelShape.SQUARE,
        grid_type=PixelGridType.REGULAR_HEX,
        name="HEXSQUARECAM",
        frame=CameraFrame(focal_length=10 * u.m),
    )
    return geom
