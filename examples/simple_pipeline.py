"""
The most basic pipeline, using no special features of the framework other 
than a for-loop. This is useful for debugging and profiling of speed.
"""

from ctapipe.instrument import CameraGeometry
from ctapipe.io.hessio import hessio_event_source
from ctapipe.calib.camera.r1 import HessioR1Calibrator
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.instrument.utils import get_camera_types
import sys
import numpy as np

if __name__ == '__main__':

    filename = sys.argv[1]

    source = hessio_event_source(filename, max_events=None,
                                 allowed_tels=np.arange(279,423))

    cal_r0 = HessioR1Calibrator(None,None)
    cal_dl0 = CameraDL0Reducer(None,None)
    cal_dl1 = CameraDL1Calibrator(None,None)

    for data in source:

        print("EVENT", data.r0.event_id)
        cal_r0.calibrate(data)
        cal_dl0.reduce(data)
        cal_dl1.calibrate(data)
        

