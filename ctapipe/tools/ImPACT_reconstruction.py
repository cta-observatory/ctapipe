from ctapipe.coordinates import *
from ctapipe.instrument import CameraGeometry
from ctapipe.image import tailcuts_clean, dilate
from ctapipe.reco.ImPACT import ImPACTFitter
import astropy.units as u
from ctapipe.core import Tool

class ImPACTRecontruction(Tool):

    def __init__(self):
        self.geoms = dict()
        self.ImPACT = ImPACTFitter("")
        return

    def reconstruct_event(self, event):

        # store MC pointing direction for the array
        array_pointing = HorizonFrame(alt = event.mcheader.run_array_direction[1]*u.rad,
                                      az = event.mcheader.run_array_direction[0]*u.rad)
        tilted_system = TiltedGroundFrame(pointing_direction=array_pointing)

        image = {}
        pixel_x = {}
        pixel_y = {}
        pixel_area = {}
        tel_type = {}
        tel_x = {}
        tel_y = {}

        for tel_id in event.dl0.tels_with_data:
            pmt_signal = event.dl1.tel[tel_id].image[0]

            nom_system = NominalFrame(array_direction=array_pointing,
                                      pointing_direction=array_pointing)

            pix_x, pix_y = event.inst.pixel_pos[tel_id]
            fl = event.inst.optical_foclen[tel_id]

            if tel_id not in self.geoms:
                self.geoms[tel_id] = CameraGeometry.guess(pix_x, pix_y,
                                        event.inst.optical_foclen[tel_id])


            # Transform the pixels positions into nominal coordinates
            camera_coord = CameraFrame(x=pix_x, y=pix_y, z=np.zeros(pix_x.shape) * u.m,
                                       focal_length=fl,
                                       rotation= -1* self.geoms[tel_id].cam_rotation)
            nom_coord = camera_coord.transform_to(nom_system)
            tx, ty, tz = event.inst.tel_pos[tel_id]

            # ImPACT reconstruction is performed in the tilted system, so we need to transform tel positions
            grd_tel = GroundFrame(x=tx * u.m, y=ty * u.m, z=tz * u.m)
            tilt_tel = grd_tel.transform_to(tilted_system)

            mask = tailcuts_clean(self.geoms[tel_id], pmt_signal, 1,
                                  picture_thresh=10,
                                  boundary_thresh=5)
            for i in range(5):
                dilate(self.geoms[tel_id], mask)

            pixel_area[tel_id] = self.geoms[tel_id].pix_area

            pixel_x[tel_id] = nom_coord.x[mask]
            pixel_y[tel_id] = nom_coord.y[mask]

            tel_x[tel_id] = tilt_tel.x
            tel_y[tel_id] = tilt_tel.y

            tel_type[tel_id] = self.geoms[tel_id].cam_id
            image[tel_id] = pmt_signal[mask]

        reco_shower = event.dl2.shower
        reco_energy = event.dl2.energy

        self.ImPACT.set_event_properties(image, pixel_x, pixel_y, pixel_area, tel_type, tel_x, tel_y, array_pointing)
        ImPACT_shower, ImPACT_energy = self.ImPACT.predict(reco_shower, reco_energy)
        event.dl2.shower = ImPACT_shower
        event.dl2.energy = ImPACT_energy