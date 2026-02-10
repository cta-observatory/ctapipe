"""
Common test fixtures for the instrument module
"""

import json
import os
import shutil

import pytest

from ctapipe.utils.download import download_file_cached
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
    Set up CTAPIPE_SVC_PATH for testing with both service data files and instrument data.

    Creates dummy array-element-ids.json and subarray-ids.json files following
    the schemas from:
    https://gitlab.cta-observatory.org/cta-computing/common/identifiers
    """
    metadata = _get_schema_metadata()

    # Create array-element-ids.json
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

    # Set CTAPIPE_SVC_PATH to include both tmp_path (for JSON) and instrument_dir (for optics/cameras)
    before = os.getenv("CTAPIPE_SVC_PATH")
    os.environ["CTAPIPE_SVC_PATH"] = f"{tmp_path}:{instrument_dir}"

    yield

    if before is None:
        del os.environ["CTAPIPE_SVC_PATH"]
    else:
        os.environ["CTAPIPE_SVC_PATH"] = before


def _get_schema_metadata():
    """
    Download and cache the JSON schemas from the CTA identifiers repository.
    Returns metadata templates from the schemas.
    """
    base_url = "https://gitlab.cta-observatory.org/cta-computing/common/identifiers/-/raw/main/"

    schemas = {
        "array_elements": "array-element-ids.schema.json",
        "subarrays": "subarray-ids.schema.json",
    }

    metadata = {}
    for name, filename in schemas.items():
        # Download and cache schema using ctapipe's download utilities
        schema_path = download_file_cached(
            filename,
            default_url=base_url,
            progress=False,
        )

        # Load schema
        with open(schema_path) as f:
            schema = json.load(f)

        # Extract metadata template from schema
        metadata[name] = {
            "$schema": schema.get("$id", ""),
            "CTA DATA MODEL URL": "https://gitlab.cta-observatory.org/cta-computing/common",
            "CTA DATA MODEL VERSION": 2,
            "CTA DATA PRODUCT DESCRIPTION": schema.get("description", ""),
            "CTA PRODUCT ID": f"test-{name}-uuid",
        }

    # Add array_elements reference for subarrays metadata
    metadata["subarrays"]["array_elements"] = "test-array_elements-uuid"

    return metadata
