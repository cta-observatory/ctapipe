"""
Common test fixtures for the instrument module
"""

import os
import shutil

import astropy.units as u
import numpy as np
import pytest

from ctapipe.utils.datasets import get_dataset_path
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
def svc_path(tmp_path, instrument_dir):
    """
    Set up CTAPIPE_SVC_PATH for testing with complete CTAO service data structure.

    Creates all required service data files following the CTAO data model format:
    - array-element-ids.json: telescope ID to name mapping (fixed schema)
    - subarray-ids.json: subarray definitions
    - positions/: ECSV files with telescope positions for each site
    - array-elements/{ae_id:03d}/: symlinks to telescope-type directories
    - array-elements/{type}/: telescope type directories with optics, camera geometry, and readout files

    The schemas follow:
    https://gitlab.cta-observatory.org/cta-computing/common/identifiers
    """
    from astropy.io import ascii
    from astropy.table import QTable

    from ctapipe.instrument.optics import ReflectorShape, SizeType

    metadata = _get_schema_metadata()

    # Create array-element-ids.json (fixed schema, no type field)
    array_element_ids = {
        "metadata": metadata["array_elements"],
        "array_elements": [
            {"id": 1, "name": "LSTN-01"},
            {"id": 2, "name": "LSTN-02"},
            {"id": 3, "name": "LSTN-03"},
            {"id": 4, "name": "LSTN-04"},
            {"id": 5, "name": "MSTN-01"},
            {"id": 6, "name": "MSTN-02"},
        ],
    }

    array_element_ids_path = tmp_path / "array-element-ids.json"
    with open(array_element_ids_path, "w") as f:
        json.dump(array_element_ids, f, indent=2)

    # Create subarray-ids.json
    subarray_ids = {
        "metadata": metadata["subarrays"],
        "subarrays": [
            {
                "id": 1,
                "name": "CTAO-N LST Subarray",
                "site": "CTAO-North",
                "array_element_ids": [1, 2, 3, 4],
            },
            {
                "id": 3,
                "name": "CTAO-N Test Array",
                "site": "CTAO-North",
                "array_element_ids": [1, 2, 5, 6],
            },
        ],
    }

    subarray_ids_path = tmp_path / "subarray-ids.json"
    with open(subarray_ids_path, "w") as f:
        json.dump(subarray_ids, f, indent=2)

    # Create positions directory and files
    positions_dir = tmp_path / "positions"
    positions_dir.mkdir()

    # Reference location for CTAO-North (La Palma) in ITRS coordinates
    from astropy import units as u
    from astropy.coordinates.earth import EarthLocation

    reference_location = EarthLocation(
        lon=-17.8920 * u.deg, lat=28.7569 * u.deg, height=2200 * u.m
    )
    itrs = reference_location.itrs

    # Create positions table for CTAO-North
    positions_north = QTable(
        {
            "ae_id": [1, 2, 3, 4, 5, 6],
            "name": ["LSTN-01", "LSTN-02", "LSTN-03", "LSTN-04", "MSTN-01", "MSTN-02"],
            "x": [0.0, 50.0, -50.0, 0.0, 100.0, -100.0] * u.m,
            "y": [0.0, 50.0, 50.0, -50.0, 0.0, 0.0] * u.m,
            "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * u.m,
        }
    )
    positions_north.meta["reference_x"] = str(itrs.x)
    positions_north.meta["reference_y"] = str(itrs.y)
    positions_north.meta["reference_z"] = str(itrs.z)
    positions_north.meta["site"] = "CTAO-North"

    positions_file = positions_dir / "CTAO-North_array_element_positions.ecsv"
    ascii.write(positions_north, positions_file, format="ecsv", overwrite=True)

    # Create array-elements directory with telescope-type subdirectories
    array_elements_dir = tmp_path / "array-elements"
    array_elements_dir.mkdir()

    # Create LSTN telescope type directory and files
    lst_dir = array_elements_dir / "LSTN"
    lst_dir.mkdir()

    # Copy existing LSTCam FITS files (source uses lowercase 'cam')
    lst_geom_path = get_dataset_path("LSTcam.camgeom.fits.gz")
    shutil.copy(lst_geom_path, lst_dir / "LSTN.camgeom.fits.gz")

    lst_readout_path = get_dataset_path("LSTcam.camreadout.fits.gz")
    shutil.copy(lst_readout_path, lst_dir / "LSTN.camreadout.fits.gz")

    # LSTN optics
    lst_optics = QTable(
        {
            "optics_name": ["LSTN"],
            "size_type": [SizeType.LST.value],
            "reflector_shape": [ReflectorShape.PARABOLIC.value],
            "n_mirrors": [1],
            "equivalent_focal_length": [28.0] * u.m,
            "effective_focal_length": [28.3] * u.m,
            "mirror_area": [386.0] * u.m**2,
            "n_mirror_tiles": [198],
        }
    )
    ascii.write(lst_optics, lst_dir / "LSTN.optics.ecsv", format="ecsv", overwrite=True)

    # Create MSTN telescope type directory and files
    mst_nectarcam_dir = array_elements_dir / "MSTN"
    mst_nectarcam_dir.mkdir()

    # Copy existing NectarCam FITS files
    nectarcam_geom_path = get_dataset_path("NectarCam.camgeom.fits.gz")
    shutil.copy(nectarcam_geom_path, mst_nectarcam_dir / "MSTN.camgeom.fits.gz")

    nectarcam_readout_path = get_dataset_path("NectarCam.camreadout.fits.gz")
    shutil.copy(
        nectarcam_readout_path,
        mst_nectarcam_dir / "MSTN.camreadout.fits.gz",
    )

    # MSTN optics
    mst_optics = QTable(
        {
            "optics_name": ["MSTN"],
            "size_type": [SizeType.MST.value],
            "reflector_shape": [ReflectorShape.DAVIES_COTTON.value],
            "n_mirrors": [1],
            "equivalent_focal_length": [16.0] * u.m,
            "effective_focal_length": [16.445] * u.m,
            "mirror_area": [88.7] * u.m**2,
            "n_mirror_tiles": [86],
        }
    )
    ascii.write(
        mst_optics,
        mst_nectarcam_dir / "MSTN.optics.ecsv",
        format="ecsv",
        overwrite=True,
    )

    # Create symlinks from ae_id to telescope type directories
    # LSTN-01, 02, 03, 04 -> LSTN
    for ae_id in [1, 2, 3, 4]:
        symlink_path = array_elements_dir / f"{ae_id:03d}"
        symlink_path.symlink_to("LSTN")

    # MSTN-01, 02 -> MSTN
    for ae_id in [5, 6]:
        symlink_path = array_elements_dir / f"{ae_id:03d}"
        symlink_path.symlink_to("MSTN")

    # Set CTAPIPE_SVC_PATH to include tmp_path and telescope type directories
    # This allows get_table_dataset to find files named LSTN.optics.ecsv, etc.
    before = os.getenv("CTAPIPE_SVC_PATH")
    search_paths = [
        str(tmp_path),
        str(lst_dir),
        str(mst_nectarcam_dir),
    ]
    os.environ["CTAPIPE_SVC_PATH"] = os.pathsep.join(search_paths)

    yield tmp_path

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
