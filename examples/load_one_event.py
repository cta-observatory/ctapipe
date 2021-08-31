"""
very simple example that loads a single event into memory, for exploration
purposes
"""
import sys

from ctapipe.calib import CameraCalibrator
from ctapipe.io import EventSource
from ctapipe.utils import get_dataset_path
from ctapipe.image import ImageProcessor
from ctapipe.reco import ShowerProcessor

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    else:
        filename = get_dataset_path("gamma_test_large.simtel.gz")

    with EventSource(filename, max_events=1) as source:
        calib = CameraCalibrator(subarray=source.subarray)
        process_images = ImageProcessor(subarray=source.subarray)
        process_shower = ShowerProcessor(subarray=source.subarray)

        for event in source:
            calib(event)
            process_images(event)
            process_shower(event)

    print(event)
