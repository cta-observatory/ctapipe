import pytest
from ctapipe.io import EventSource, DataWriter
from ctapipe.utils import get_dataset_path


@pytest.fixture(scope="session")
def r1_path(tmp_path_factory):
    return tmp_path_factory.mktemp("r1")


@pytest.fixture(scope="session")
def r1_hdf5_file(r1_path):
    source = EventSource(
        get_dataset_path("gamma_LaPalma_baseline_20Zd_180Az_prod3b_test.simtel.gz"),
        max_events=5,
        allowed_tels=[1, 2, 3, 4],
    )

    path = r1_path / "test_r1.h5"

    writer = DataWriter(
        event_source=source,
        output_path=path,
        write_parameters=False,
        write_images=False,
        write_stereo_shower=False,
        write_mono_shower=False,
        write_raw_waveforms=False,
        write_waveforms=True,
    )

    for e in source:
        writer(e)

    writer.finish()

    return path
