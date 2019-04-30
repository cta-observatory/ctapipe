"""
very simple example that loads a single event into memory, for exploration
purposes
"""
import sys

from ctapipe.calib import CameraCalibrator
from ctapipe.io import event_source
from ctapipe.utils import get_dataset_path

if __name__ == '__main__':

    calib = CameraCalibrator()

    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    else:
        filename = get_dataset_path("gamma_test_large.simtel.gz")

    with event_source(filename, max_events=1) as source:
        for event in source:
            calib(event)

    print(event)
