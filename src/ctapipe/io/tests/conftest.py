import pytest

from ctapipe.io import DataWriter, EventSource


@pytest.fixture(scope="session")
def r1_path(tmp_path_factory):
    return tmp_path_factory.mktemp("r1")


@pytest.fixture(scope="session")
def r1_hdf5_file(prod5_proton_simtel_path, r1_path):
    source = EventSource(
        prod5_proton_simtel_path,
        max_events=5,
    )

    path = r1_path / "test_r1.h5"

    writer = DataWriter(
        event_source=source,
        output_path=path,
        write_dl1_parameters=False,
        write_dl1_images=False,
        write_dl2=False,
        write_r0_waveforms=False,
        write_r1_waveforms=True,
    )

    for e in source:
        writer(e)

    writer.finish()

    return path
