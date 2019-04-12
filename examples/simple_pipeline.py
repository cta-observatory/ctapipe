"""
The most basic pipeline, using no special features of the framework other 
than a for-loop. This is useful for debugging and profiling of speed.
"""

import sys

from ctapipe.calib import CameraCalibrator
from ctapipe.io import event_source

if __name__ == '__main__':

    filename = sys.argv[1]

    source = event_source(filename, max_events=None)

    cal = CameraCalibrator()

    for ii, event in enumerate(source):

        print(
            "{} EVENT_ID: {}, ENERGY: {:.2f}, NTELS:{}".format(
                ii,
                event.r0.event_id, event.mc.energy, len(event.dl0.tels_with_data)
            )
        )

        cal.calibrate(event)

        # now the calibrated images are in data.dl1.tel[x].image
