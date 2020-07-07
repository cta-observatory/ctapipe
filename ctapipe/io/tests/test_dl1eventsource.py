from ctapipe.utils import get_dataset_path
from ctapipe.io import DataLevel
from ctapipe.io.dl1eventsource import DL1EventSource
import subprocess
import tempfile

# generate a testfile, that contains everything we want to test
filepath = get_dataset_path("gamma_test_large.simtel.gz")
tempdir = tempfile.TemporaryDirectory("temp")
testfile = f"{tempdir.name}/testfile.dl1.h5"
command = f"ctapipe-stage1-process --input {filepath} --output {testfile} --write-parameters --write-images --max-events 20 --allowed-tels=[1,2,3]"
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
#testfile_only_images = f"{tempdir.name}/testfile.dl1.h5"
#command = f"ctapipe-stage1-process --input {filepath} --output {testfile_only_images} --write-images --max-events 20 --allowed-tels=[1,2,3]"
#process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

#testfile_only_params = f"{tempdir.name}/testfile.dl1.h5"
#command = f"ctapipe-stage1-process --input {filepath} --output {testfile_only_params} --write-parameters --max-events 20 --allowed-tels=[1,2,3]"
#process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)



def test_metadata():
    with DL1EventSource(input_url=testfile) as source:
        assert source.is_simulation
        assert source.datalevels == (
            DataLevel.DL1_IMAGES,
            DataLevel.DL1_PARAMETERS
        )
        assert list(source.obs_ids) == [7514]
        assert source.mc_headers[7514].corsika_version == 6990


def test_max_events():
    max_events = 5
    with DL1EventSource(input_url=testfile, max_events=max_events) as source:
        count = 0
        assert source.max_events == max_events
        for _ in source:
            count += 1
        assert count == max_events


def test_allowed_tels():
    allowed_tels = {1, 2}
    with DL1EventSource(input_url=testfile, allowed_tels=allowed_tels) as source:
        assert source.allowed_tels == allowed_tels
        for event in source:
            assert event.trigger.tels_with_trigger.issubset(allowed_tels)
            for tel in event.dl1.tel:
                assert tel in allowed_tels


def test_dl1_data():
    with DL1EventSource(input_url=testfile) as source:
        for event in source:
            for tel in event.dl1.tel:
                assert event.dl1.tel[tel].image.any()

def test_only_images():
    with DL1EventSource(input_url=testfile) as source:
        for event in source:
            for tel in event.dl1.tel:
                assert event.dl1.tel[tel].image.any()
                assert event.dl1.tel[tel].hillas.x == np.nan
