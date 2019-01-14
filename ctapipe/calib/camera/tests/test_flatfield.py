from ctapipe.utils import get_dataset_path
from ctapipe.io.nectarcameventsource import NectarCAMEventSource
from ctapipe.calib.camera.flatfield import FlasherFlatFieldCalculator


def test_flasherflatfieldcalculator():

    example_file_path = get_dataset_path("NectarCAM.Run0890.10events.fits.fz")

    inputfile_reader = NectarCAMEventSource(
        input_url=example_file_path,
        max_events=10
    )

    ff_calculator = FlasherFlatFieldCalculator(sample_size=3, tel_id=0)

    for event in inputfile_reader:

        ff_data = ff_calculator.calculate_relative_gain(event)

        if ff_calculator.num_events_seen == ff_calculator.sample_size:
            assert ff_data
