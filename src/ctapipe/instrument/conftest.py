"""
Common test fixtures for the instrument module
"""

import json
import os
import shutil

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates.earth import EarthLocation
from astropy.io import ascii
from astropy.table import QTable

from ctapipe.core import run_tool
from ctapipe.instrument.optics import ReflectorShape, SizeType
from ctapipe.io import metadata as meta
from ctapipe.tools.dump_instrument import DumpInstrumentTool
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.utils.filelock import FileLock

# ---------------------------------------------------------------------------
# Private helpers shared between svc_path fixtures
# ---------------------------------------------------------------------------

_CTAO_IDENTIFIERS_URL = (
    "https://gitlab.cta-observatory.org/cta-computing/common/identifiers"
)


def _make_ids_reference(description, data_model_name, site="CTAO-North"):
    """Create a Reference metadata object for CTAO identifier JSON files."""

    return meta.Reference(
        contact=meta.Contact(),
        product=meta.Product(
            description=description,
            data_category="Other",
            data_association="Subarray",
            data_model_name=data_model_name,
            data_model_version="2.0",
            data_model_url=_CTAO_IDENTIFIERS_URL,
            format="json",
        ),
        process=meta.Process(),
        activity=meta.Activity(),
        instrument=meta.Instrument(site=site),
    )


def _write_instrument_meta(tmp_path, site="CTAO-North"):
    """Write instrument.meta.json into *tmp_path*."""

    ref = meta.Reference(
        contact=meta.Contact(),
        product=meta.Product(
            description="Instrument description service data for CTAO",
            data_category="Other",
            data_association="Subarray",
            data_model_name="CTAO Service Data",
            data_model_version="2.0",
            data_model_url="",
            format="json",
        ),
        process=meta.Process(),
        activity=meta.Activity(),
        instrument=meta.Instrument(site=site, class_="Subarray"),
    )
    with open(tmp_path / "instrument.meta.json", "w") as f:
        json.dump(ref.to_dict(), f, indent=2)


def _write_array_element_ids(tmp_path, array_elements, site="CTAO-North"):
    """Write array-element-ids.json into *tmp_path*."""
    ref = _make_ids_reference(
        description="Array element IDs for CTAO",
        data_model_name="ctao.common.identifiers.array_elements",
        site=site,
    )
    data = {"metadata": ref.to_dict(), "array_elements": array_elements}
    with open(tmp_path / "array-element-ids.json", "w") as f:
        json.dump(data, f, indent=2)


def _write_subarray_ids(tmp_path, subarrays, site="CTAO-North"):
    """Write subarray-ids.json into *tmp_path*."""
    ref = _make_ids_reference(
        description="Subarray IDs for CTAO",
        data_model_name="ctao.common.identifiers.subarrays",
        site=site,
    )
    data = {"metadata": ref.to_dict(), "subarrays": subarrays}
    with open(tmp_path / "subarray-ids.json", "w") as f:
        json.dump(data, f, indent=2)


def _write_positions_file(
    positions_dir, ae_ids, names, x_m, y_m, z_m, site="CTAO-North"
):
    """Write an ECSV positions file for the given array elements into *positions_dir*."""
    itrs = EarthLocation(
        lon=-17.8920 * u.deg, lat=28.7569 * u.deg, height=2200 * u.m
    ).itrs

    positions = QTable(
        {
            "ae_id": ae_ids,
            "name": names,
            "x": x_m * u.m,
            "y": y_m * u.m,
            "z": z_m * u.m,
        }
    )
    positions.meta["reference_x"] = str(itrs.x)
    positions.meta["reference_y"] = str(itrs.y)
    positions.meta["reference_z"] = str(itrs.z)
    positions.meta["site"] = site

    ascii.write(
        positions,
        positions_dir / f"{site}_ArrayElementPositions.ecsv",
        format="ecsv",
        overwrite=True,
    )


# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def instrument_dir(tmp_path_factory):
    """Dump instrument of prod5 subarray into a directory as fits"""
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
def svc_path(tmp_path, instrument_dir, monkeypatch):
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
    site = "CTAO-North"

    _write_instrument_meta(tmp_path, site)

    _write_array_element_ids(
        tmp_path,
        array_elements=[
            {"id": 1, "name": "LSTN-01"},
            {"id": 2, "name": "LSTN-02"},
            {"id": 3, "name": "LSTN-03"},
            {"id": 4, "name": "LSTN-04"},
            {"id": 5, "name": "MSTN-01"},
            {"id": 6, "name": "MSTN-02"},
        ],
        site=site,
    )

    _write_subarray_ids(
        tmp_path,
        subarrays=[
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
        site=site,
    )

    # Create positions directory and files
    positions_dir = tmp_path / "positions"
    positions_dir.mkdir()

    _write_positions_file(
        positions_dir,
        ae_ids=[1, 2, 3, 4, 5, 6],
        names=["LSTN-01", "LSTN-02", "LSTN-03", "LSTN-04", "MSTN-01", "MSTN-02"],
        x_m=np.array([0.0, 50.0, -50.0, 0.0, 100.0, -100.0]),
        y_m=np.array([0.0, 50.0, 50.0, -50.0, 0.0, 0.0]),
        z_m=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        site=site,
    )

    # Create array-elements directory with telescope-type subdirectories
    array_elements_dir = tmp_path / "array-elements"
    array_elements_dir.mkdir()

    # Create LSTN telescope type directory and files
    lst_dir = array_elements_dir / "LSTN"
    lst_dir.mkdir()

    # Copy existing LSTcam FITS files (source uses lowercase 'cam')
    lst_geom_path = get_dataset_path("LSTcam.camgeom.fits.gz")
    shutil.copy(lst_geom_path, lst_dir / "LSTN.camgeom.fits.gz")

    lst_readout_path = get_dataset_path("LSTcam.camreadout.fits.gz")
    shutil.copy(lst_readout_path, lst_dir / "LSTN.camreadout.fits.gz")

    # LSTN optics (empty table with metadata)
    lst_optics = QTable()
    lst_optics.meta["TAB_VER"] = "4.0"
    lst_optics.meta["optics_name"] = "LSTN"
    lst_optics.meta["size_type"] = SizeType.LST.value
    lst_optics.meta["reflector_shape"] = ReflectorShape.PARABOLIC.value
    lst_optics.meta["n_mirrors"] = 1
    lst_optics.meta["equivalent_focal_length"] = str(28.0 * u.m)
    lst_optics.meta["effective_focal_length"] = str(28.3 * u.m)
    lst_optics.meta["mirror_area"] = str(386.0 * u.m**2)
    lst_optics.meta["n_mirror_tiles"] = 198
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

    # MSTN optics (empty table with metadata)
    mst_optics = QTable()
    mst_optics.meta["TAB_VER"] = "4.0"
    mst_optics.meta["optics_name"] = "MSTN"
    mst_optics.meta["size_type"] = SizeType.MST.value
    mst_optics.meta["reflector_shape"] = ReflectorShape.DAVIES_COTTON.value
    mst_optics.meta["n_mirrors"] = 1
    mst_optics.meta["equivalent_focal_length"] = str(16.0 * u.m)
    mst_optics.meta["effective_focal_length"] = str(16.445 * u.m)
    mst_optics.meta["mirror_area"] = str(88.7 * u.m**2)
    mst_optics.meta["n_mirror_tiles"] = 86
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

    # Set CTAPIPE_SVC_PATH to include tmp_path, positions directory,
    # telescope type directories, and instrument_dir (for legacy from_name methods)
    # This allows get_table_dataset to find files named LSTN.optics.ecsv, etc.
    # and also the legacy optics.fits.gz and camera files
    search_paths = [
        str(tmp_path),
        str(positions_dir),
        str(lst_dir),
        str(mst_nectarcam_dir),
        str(instrument_dir),
    ]
    monkeypatch.setenv("CTAPIPE_SVC_PATH", os.pathsep.join(search_paths))

    yield tmp_path


@pytest.fixture(scope="function")
def svc_path_aeid_specific(tmp_path, instrument_dir, monkeypatch):
    """
    Set up CTAPIPE_SVC_PATH with ae_id-specific files (no deduplication).

    This fixture creates a service data structure where each telescope has
    its own configuration files named with ae_id (e.g., 001.optics.ecsv)
    instead of shared telescope-type files (e.g., LSTN.optics.ecsv).

    This tests the fallback mechanism: files are first searched as {ae_id}.{type},
    then as {tel_type}.{type} if not found.
    """
    site = "CTAO-North"

    _write_instrument_meta(tmp_path, site)

    _write_array_element_ids(
        tmp_path,
        array_elements=[
            {"id": 1, "name": "LSTN-01"},
            {"id": 2, "name": "LSTN-02"},
        ],
        site=site,
    )

    _write_subarray_ids(
        tmp_path,
        subarrays=[
            {
                "id": 1,
                "name": "CTAO-N Test Subarray",
                "site": "CTAO-North",
                "array_element_ids": [1, 2],
            },
        ],
        site=site,
    )

    # Create positions directory
    positions_dir = tmp_path / "positions"
    positions_dir.mkdir()

    _write_positions_file(
        positions_dir,
        ae_ids=[1, 2],
        names=["LSTN-01", "LSTN-02"],
        x_m=np.array([0.0, 50.0]),
        y_m=np.array([0.0, 50.0]),
        z_m=np.array([0.0, 0.0]),
        site=site,
    )

    # Create array-elements directory
    array_elements_dir = tmp_path / "array-elements"
    array_elements_dir.mkdir()

    # Create LSTN telescope type directory (for symlink resolution)
    lst_dir = array_elements_dir / "LSTN"
    lst_dir.mkdir()

    # For telescope 001: create ae_id-specific files with custom optics
    ae_001_dir = array_elements_dir / "001"
    ae_001_dir.symlink_to("LSTN")

    # Create 001.optics with slightly different parameters (empty table with metadata)
    optics_001 = QTable()
    optics_001.meta["TAB_VER"] = "4.0"
    optics_001.meta["optics_name"] = "LSTN-01-Custom"
    optics_001.meta["size_type"] = SizeType.LST.value
    optics_001.meta["reflector_shape"] = ReflectorShape.PARABOLIC.value
    optics_001.meta["n_mirrors"] = 1
    optics_001.meta["equivalent_focal_length"] = str(28.0 * u.m)
    optics_001.meta["effective_focal_length"] = str(
        28.5 * u.m
    )  # Different from default
    optics_001.meta["mirror_area"] = str(390.0 * u.m**2)  # Different from default
    optics_001.meta["n_mirror_tiles"] = 198
    ascii.write(optics_001, lst_dir / "001.optics.ecsv", format="ecsv", overwrite=True)

    # Copy camera files with 001 prefix
    lst_geom_path = get_dataset_path("LSTcam.camgeom.fits.gz")
    shutil.copy(lst_geom_path, lst_dir / "001.camgeom.fits.gz")

    lst_readout_path = get_dataset_path("LSTcam.camreadout.fits.gz")
    shutil.copy(lst_readout_path, lst_dir / "001.camreadout.fits.gz")

    # For telescope 002: create symlink but use shared telescope-type files as fallback
    ae_002_dir = array_elements_dir / "002"
    ae_002_dir.symlink_to("LSTN")

    # Create shared LSTN files (002 will fall back to these, empty table with metadata)
    lst_optics = QTable()
    lst_optics.meta["TAB_VER"] = "4.0"
    lst_optics.meta["optics_name"] = "LSTN"
    lst_optics.meta["size_type"] = SizeType.LST.value
    lst_optics.meta["reflector_shape"] = ReflectorShape.PARABOLIC.value
    lst_optics.meta["n_mirrors"] = 1
    lst_optics.meta["equivalent_focal_length"] = str(28.0 * u.m)
    lst_optics.meta["effective_focal_length"] = str(28.3 * u.m)
    lst_optics.meta["mirror_area"] = str(386.0 * u.m**2)
    lst_optics.meta["n_mirror_tiles"] = 198
    ascii.write(lst_optics, lst_dir / "LSTN.optics.ecsv", format="ecsv", overwrite=True)

    shutil.copy(lst_geom_path, lst_dir / "LSTN.camgeom.fits.gz")
    shutil.copy(lst_readout_path, lst_dir / "LSTN.camreadout.fits.gz")

    # Set CTAPIPE_SVC_PATH
    search_paths = [
        str(tmp_path),
        str(positions_dir),
        str(lst_dir),
        str(instrument_dir),
    ]
    monkeypatch.setenv("CTAPIPE_SVC_PATH", os.pathsep.join(search_paths))

    yield tmp_path
