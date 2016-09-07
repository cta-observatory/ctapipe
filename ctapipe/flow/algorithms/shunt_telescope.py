from ctapipe.utils.datasets import get_path
import ctapipe.instrument.InstrumentDescription as ID
from ctapipe.core import Component
from ctapipe.io import CameraGeometry
from ctapipe.io.files import InputFile
from traitlets import Unicode


class ShuntTelescope(Component):
    """ShuntTelescope class represents a Stage for pipeline.
        It shunts event based on telescope type
    """
    def __init__(self,parent,config):
        Component.__init__(self,parent)
        self.geom_dict = dict()

    def init(self):
        self.log.info("--- ShuntTelescope init ---")
        return True

    def run(self, event):
        triggered_telescopes = event.dl0.tels_with_data
        for tel_id in triggered_telescopes:
            cam_dimensions = (event.dl0.tel[tel_id].num_pixels,
                                  event.meta.optical_foclen[tel_id])
            if not cam_dimensions in self.geom_dict:
                geom = CameraGeometry.guess(*event.meta.pixel_pos[tel_id],
                                event.meta.optical_foclen[tel_id])
                self.geom_dict[cam_dimensions] = geom
            camera_id = self.geom_dict[cam_dimensions].cam_id
            if camera_id == 'LSTCam':
                return (event,'LSTCalibration')


    def finish(self):
        self.log.info("--- ShuntTelescope finish ---")
        pass
