"""
very simple example that loads a single event into memory, for exploration
purposes
"""
import sys
from ctapipe.io import event_source
from ctapipe.utils import get_dataset

from ctapipe.calib import CameraCalibrator

if __name__ == '__main__':

    calib = CameraCalibrator(r1_product="HESSIOR1Calibrator")

    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    else:
        filename = get_dataset("gamma_test_large.simtel.gz")

    with event_source(filename, max_events=1) as source:
        for event in source:
            calib.calibrate(event)

    print(event)
