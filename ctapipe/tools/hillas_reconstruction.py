import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy

import astropy.units as u
from traitlets import Dict, List, Unicode, Int, Bool
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

from ctapipe.reco.energy_reco_mva import EnergyReconstructorMVA
from ctapipe.image import FullIntegrator

from ctapipe.plotting.event_viewer import EventViewer
from astropy.table import Table

from ctapipe.reco.hillas_intersection import HillasIntersection

class EventReconstructionHillas(Tool):
    """

    """
    description = "EventReconstructionHillas"
    name = 'ctapipe-hillas-reco'

    infile = Unicode(help='input simtelarray file').tag(config=True)

    outfile = Unicode(help='output fits table').tag(config=True)

    telescopes = List(Int, None, allow_none=True,
                      help='Telescopes to include from the event file. '
                           'Default = All telescopes').tag(config=True)


    aliases = Dict(dict(infile='EventReconstructionHillas.infile',
                        outfile='EventReconstructionHillas.outfile',
                        telescopes='EventReconstructionHillas.telescopes'))

    def setup(self):

        self.geoms = dict()
        self.amp_cut = {"LSTCam": 100,
                        "NectarCam": 100,
                        "FlashCam": 100,
                        "GCT": 50}

        self.dist_cut = {"LSTCam": 2. * u.deg,
                         "NectarCam": 3.3 * u.deg,
                         "FlashCam": 3. * u.deg,
                         "GCT": 3.8 * u.deg}

        self.tail_cut = {"LSTCam": (8, 16),
                         "NectarCam": (7, 14),
                         "FlashCam": (7, 14),
                         "GCT": (3, 6)}

        # Calibrators set to default for now
        self.r1 = HessioR1Calibrator(None, None)
        self.dl0 = CameraDL0Reducer(None, None)
        self.calibrator = CameraDL1Calibrator(None, None,
                                              extractor=FullIntegrator(None, None))
        print(self.infile)
        # If we don't set this just use everything
        if len(self.telescopes) < 2:
            self.telescopes = None
        self.source = hessio_event_source(self.infile, allowed_tels=self.telescopes)

        self.fit = HillasIntersection()
        self.energy_reco = EnergyReconstructorMVA()

        self.viewer = EventViewer(draw_hillas_planes=True)

        self.output = Table(names=['EVENT_ID', 'RECO_ALT', 'RECO_AZ',
                                   'RECO_EN', 'SIM_ALT', 'SIM_AZ', 'SIM_EN', 'NTELS'],
                            dtype=[np.int64, np.float64, np.float64,
                                   np.float64, np.float64, np.float64, np.float64,
                                   np.int16])

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

        tel_type = {}
        tel_x = {}
        tel_y = {}

        hillas = {}
        hillas_nom = {}


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
            grd_tel = GroundFrame(x=tx, y=ty, z=tz)
            tilt_tel = grd_tel.transform_to(tilted_system)

            # Clean image using split level cleaning
            mask = tailcuts_clean(self.geoms[tel_id], pmt_signal,
                                  picture_thresh=self.tail_cut[self.geoms[
                                      tel_id].cam_id][1],
                                  boundary_thresh=self.tail_cut[self.geoms[
                                      tel_id].cam_id][0])

            # Perform Hillas parameterisation
            moments = None
            try:
                moments_cam = hillas_parameters(event.inst.pixel_pos[tel_id][0],
                                                event.inst.pixel_pos[tel_id][1],
                                                pmt_signal * mask)

                moments = hillas_parameters(nom_coord.x, nom_coord.y, pmt_signal * mask)

            except HillasParameterizationError as e:
                print(e)
                continue

            # Make cut based on Hillas parameters
            if self.preselect(moments, tel_id):

                # Dialte around edges of image
                for i in range(2):
                    dilate(self.geoms[tel_id], mask)

                # Save everything in dicts for reconstruction later
                tel_x[tel_id] = tilt_tel.x
                tel_y[tel_id] = tilt_tel.y

                tel_type[tel_id] = self.geoms[tel_id].cam_id

                hillas[tel_id] = moments_cam
                hillas_nom[tel_id] = moments

        # Cut on number of telescopes remaining
        if len(tel_x) > 1:

            fit_result = self.fit.predict(hillas_nom, tel_x, tel_y, array_pointing)

            energy_result = self.energy_reco.predict(fit_result, hillas_nom, tel_type,
                                                     tel_x, tel_y, array_pointing)
            print(fit_result, energy_result)

            # insert the row into the table
            self.output.add_row((event.dl0.event_id, fit_result.alt,
                                 fit_result.az, energy_result.energy,
                                 event.mc.alt, event.mc.az, event.mc.energy, len(tel_x)))



def main():
    exe = EventReconstructionHillas()

    exe.run()


if __name__ == '__main__':
    main()