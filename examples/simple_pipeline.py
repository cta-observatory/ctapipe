"""
The most basic pipeline, using no special features of the framework other 
than a for-loop. This is useful for debugging and profiling of speed.
"""

import sys

from ctapipe.calib import CameraCalibrator
from ctapipe.io import event_source

if __name__ == '__main__':

    filename = sys.argv[1]
    try:
        max_events = int(sys.argv[2])
    except IndexError:
        max_events = None

    source = event_source(filename, max_events=max_events)

    cal = CameraCalibrator(r1_product="HESSIOR1Calibrator")

    for data in source:

        print(
            "EVENT: {}, ENERGY: {:.2f}, TELS:{}".format(
                data.r0.event_id, data.mc.energy, len(data.dl0.tels_with_data)
            )
        )

        cal.calibrate(data)

        # now the calibrated images are in data.dl1.tel[x].image
