"""
very simple example that loads a single event into memory, for exploration
purposes
"""
import sys
from ctapipe.io.hessio import hessio_event_source

from ctapipe.calib import CameraCalibrator

if __name__ == '__main__':

    calib = CameraCalibrator(None,None)

    filename = sys.argv[1]

    source = hessio_event_source(filename, max_events=1)
    event = next(source)
    calib.calibrate(event)
    del source # just to clean up and close files

    print(event)
