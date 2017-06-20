"""
The most basic pipeline, using no special features of the framework other 
than a for-loop. This is useful for debugging and profiling of speed.
"""

import sys

import numpy as np

from ctapipe.calib import CameraCalibrator
from ctapipe.io.hessio import hessio_event_source

if __name__ == '__main__':

    filename = sys.argv[1]

    source = hessio_event_source(filename, max_events=None)

    cal = CameraCalibrator(None, None)

    for data in source:

        print("EVENT: {}, ENERGY: {:.2f}, TELS:{}"
              .format(data.r0.event_id,
                      data.mc.energy,
                      len(data.dl0.tels_with_data))
              )

        cal.calibrate(data)

        # now the calibrated images are in data.dl1.tel[x].image
