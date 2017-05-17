import matplotlib
matplotlib.use('Qt5Agg')

import astropy.units as u
from traitlets import Dict, List, Unicode, Int
import numpy as np

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import HessioR1Calibrator

from ctapipe.coordinates import *
from ctapipe.core import Tool
from ctapipe.image import tailcuts_clean, dilate
from ctapipe.instrument import CameraGeometry
from ctapipe.image import hillas_parameters, HillasParameterizationError
from ctapipe.io.hessio import hessio_event_source

from astropy.table import Table


class GetTrainingParams(Tool):
    """

    """
    description = "GetTrainingParams"
    name = 'ctapipe-get-training-params'

    infile = Unicode(help='input simtelarray file').tag(config=True)

    outfile = Unicode(help='output fits table').tag(config=True)

    telescopes = List(Int, None, allow_none=True,
                      help='Telescopes to include from the event file. '
                           'Default = All telescopes').tag(config=True)

    aliases = Dict(dict(infile='GetTrainingParams.infile',
                        outfile='GetTrainingParams.outfile',
                        telescopes='GetTrainingParams.telescopes'))

    def setup(self):

        self.geoms = dict()
        self.amp_cut = {"LSTCam": 100,
                        "NectarCam": 100,
                        "FlashCam": 100,
                        "GCT": 50,
                        "SST-1m": 50,
                        "ASTRI": 50,
                        "SCTCam": 100
                        }

        self.dist_cut = {"LSTCam": 2. * u.deg,
                         "NectarCam": 3.3 * u.deg,
                         "FlashCam": 3.3 * u.deg,
                         "GCT": 3.8 * u.deg,
                         "SST-1m": 3.8 * u.deg,
                         "ASTRI": 3.8 * u.deg,
                         "SCTCam": 3.3 * u.deg
                         }

        self.tail_cut = {"LSTCam": (8, 16),
                         "NectarCam": (7, 14),
                         "FlashCam": (7, 14),
                         "GCT": (3, 6),
                         "SST-1m": (3, 6),
                         "ASTRI": (3, 6),
                         "SCTCam": (3, 6)
                         }

        # Calibrators set to default for now
        self.r1 = HessioR1Calibrator(None, None)
        self.dl0 = CameraDL0Reducer(None, None)
        self.calibrator = CameraDL1Calibrator(None, None)

        # If we don't set this just use everything
        if len(self.telescopes) < 2:
            self.telescopes = None
        self.source = hessio_event_source(self.infile, allowed_tels=self.telescopes)

        self.output = Table(names=['EVENT_ID', 'TEL_TYPE', 'AMP', 'WIDTH', 'LENGTH',
                                   'SIM_EN', 'IMPACT'],
                            dtype=[np.int64, np.str, np.float64, np.float64,
                                   np.float64, np.float64, np.float64])

    def start(self):

        for event in self.source:
            self.calibrate_event(event)
            self.reconstruct_event(event)

    def finish(self):
        self.output.write(self.outfile)

        return True

    def calibrate_event(self, event):
        """
        Run standard calibrators to get from r0 to dl1

        Parameters
        ----------
        event: ctapipe event container

        Returns
        -------
            None
        """
        self.r1.calibrate(event)
        self.dl0.reduce(event)
        self.calibrator.calibrate(event)  # calibrate the events

    def preselect(self, hillas, tel_id):
        """
        Perform pre-selection of telescopes (before reconstruction) based on Hillas
        Parameters

        Parameters
        ----------
        hillas: ctapipe Hillas parameter object
        tel_id: int
            Telescope ID number

        Returns
        -------
            bool: Indicate whether telescope passes cuts
        """
        if hillas is None:
            return False

        # Calculate distance of image centroid from camera centre
        dist = np.sqrt(hillas.cen_x * hillas.cen_x + hillas.cen_y * hillas.cen_y)

        # Cut based on Hillas amplitude and nominal distance
        if hillas.size > self.amp_cut[self.geoms[tel_id].cam_id] and dist < \
                self.dist_cut[self.geoms[tel_id].cam_id] and \
                        hillas.width > 0 * u.deg:
            return True

        return False

    def reconstruct_event(self, event):
        """
        Perform full event reconstruction, including Hillas and ImPACT analysis.

        Parameters
        ----------
        event: ctapipe event container

        Returns
        -------
            None
        """
        # store MC pointing direction for the array
        array_pointing = HorizonFrame(alt=event.mcheader.run_array_direction[1] * u.rad,
                                      az=event.mcheader.run_array_direction[0] * u.rad)
        tilted_system = TiltedGroundFrame(pointing_direction=array_pointing)
        mc_ground = GroundFrame(x=event.mc.core_x, y=event.mc.core_y,
                                z=0 * u.m)

        for tel_id in event.dl0.tels_with_data:
            # Get calibrated image (low gain channel only)
            pmt_signal = event.dl1.tel[tel_id].image[0]

            # Create nominal system for the telescope (this should later used telescope
            # pointing)
            nom_system = NominalFrame(array_direction=array_pointing,
                                      pointing_direction=array_pointing)

            # Create camera system of all pixels
            pix_x, pix_y = event.inst.pixel_pos[tel_id]
            fl = event.inst.optical_foclen[tel_id]
            if tel_id not in self.geoms:
                self.geoms[tel_id] = CameraGeometry.guess(pix_x, pix_y,
                                                          event.inst.optical_foclen[
                                                              tel_id])

            # Transform the pixels positions into nominal coordinates
            camera_coord = CameraFrame(x=pix_x, y=pix_y, z=np.zeros(pix_x.shape) * u.m,
                                       focal_length=fl,
                                       rotation=-1 * self.geoms[tel_id].cam_rotation)
            nom_coord = camera_coord.transform_to(nom_system)
            tx, ty, tz = event.inst.tel_pos[tel_id]

            # ImPACT reconstruction is performed in the tilted system,
            # so we need to transform tel positions
            grd_tel = GroundFrame(x=tx * u.m, y=ty * u.m, z=tz * u.m)
            tilt_tel = grd_tel.transform_to(tilted_system)

            # Clean image using split level cleaning
            mask = tailcuts_clean(self.geoms[tel_id], pmt_signal,
                                  picture_thresh=self.tail_cut[self.geoms[
                                      tel_id].cam_id][1],
                                  boundary_thresh=self.tail_cut[self.geoms[
                                      tel_id].cam_id][0])

            # Perform Hillas parameterisation
            try:
                moments = hillas_parameters(nom_coord.x, nom_coord.y, pmt_signal * mask)

            except HillasParameterizationError as e:
                print(e)
                continue

            # Make cut based on Hillas parameters
            if self.preselect(moments, tel_id):
                impact_dist = tilt_tel.separation(mc_ground)
                self.output.add_row((event.dl0.event_id, self.geoms[tel_id].cam_id,
                                     moments.size, moments.width.value, moments.length.value,
                                     event.mc.energy.value, impact_dist.value))


def main():
    tool = GetTrainingParams()
    tool.run()

if __name__ == '__main__':
    main()