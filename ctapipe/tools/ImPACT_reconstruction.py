import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy

import astropy.units as u
from traitlets import Dict, List, Unicode
import numpy as np

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import HessioR1Calibrator

from ctapipe.coordinates import *
from ctapipe.core import Tool
from ctapipe.image import tailcuts_clean, dilate
from ctapipe.instrument import CameraGeometry
from ctapipe.reco.ImPACT import ImPACTFitter
from ctapipe.image import hillas_parameters, HillasParameterizationError
from ctapipe.io.hessio import hessio_event_source
from ctapipe.io.containers import ReconstructedShowerContainer, ReconstructedEnergyContainer

from ctapipe.reco.FitGammaHillas import FitGammaHillas

from ctapipe.plotting.event_viewer import EventViewer
from astropy.table import Table


class ImPACTReconstruction(Tool):
    """
    
    """
    description = "ImPACTReco"
    name='ctapipe-ImPACT-reco'

    infile = Unicode(None, allow_none=True,
                     help='input simtelarray file').tag(config=True)

    flags = Dict(dict(hillas=({'ImPACTReco': {'only_hillas': False}},
                                   'Only perform Hillas event reconstruction')))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.geoms = dict()
        self.ImPACT = None

        self.amp_cut = None
        self.dist_cut = None
        self.tail_cut = None

        self.r1 = None
        self.dl0 = None
        self.calibrator = None

        self.source = None
        self.output = None

    def setup(self):

        self.amp_cut = {"LSTCam": 100,
                        "NectarCam": 100,
                        "FlashCam": 100,
                        "GCT": 50}

        self.dist_cut = {"LSTCam": 2. * u.deg,
                         "NectarCam": 3. * u.deg,
                         "FlashCam": 3. * u.deg,
                         "GCT": 4. * u.deg}

        self.tail_cut = {"LSTCam": (7, 14),
                         "NectarCam": (7, 14),
                         "FlashCam": (7, 14),
                         "GCT": (3, 6)}

        # Calibrators set to default for now
        self.r1 = HessioR1Calibrator(None, None)
        self.dl0 = CameraDL0Reducer(None, None)
        self.calibrator = CameraDL1Calibrator(None, None)

        # Get source ready to loop over the file
        self.infile = "/Users/dparsons/Desktop/gamma_20deg_180deg_run9021___cta-prod3-merged_desert-2150m-Paranal-subarray-3.simtel.gz"
        HB4 = [1, 2, 3, 71, 72, 73, 74, 75, 76, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
               89, 90, 104, 105, 106, 107, 108, 109,
               279, 280, 281, 282, 283, 284, 286, 287, 289, 297, 298, 299, 300, 301, 302,
               303, 304, 305, 306, 307, 308, 315,
               316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330,
               331, 332, 333, 334, 335, 336, 337,
               338, 345, 346, 347, 348, 349, 350, 375, 376, 377, 378, 379, 380, 393, 400,
               402, 403, 404, 405, 406, 408, 410,
               411, 412, 413, 414, 415, 416, 417]
        self.source = hessio_event_source(self.infile, allowed_tels=HB4)

        self.fit = FitGammaHillas()
        self.ImPACT = ImPACTFitter("")
        self.viewer = EventViewer(draw_hillas_planes=True)

        self.output = Table(names=['EVENT_ID', 'RECO_ALT', 'RECO_AZ',
                                   'RECO_ENERGY', 'SIM_ALT', 'SIM_AZ', 'SIM_EN'],
                            dtype=[np.int64, np.float64, np.float64,
                                   np.float64, np.float64, np.float64, np.float64])



    def start(self):

        for event in self.source:
            self.calibrate_event(event)
            self.reconstruct_event(event)

    def finish(self):
        self.output.write("out.fits")

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
                self.dist_cut[self.geoms[tel_id].cam_id]:
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

        hillas = {}
        hillas_nom = {}

        tel_phi = {}
        tel_theta = {}

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
                                        event.inst.optical_foclen[tel_id])

            # Transform the pixels positions into nominal coordinates
            camera_coord = CameraFrame(x=pix_x, y=pix_y, z=np.zeros(pix_x.shape) * u.m,
                                       focal_length=fl,
                                       rotation= -1* self.geoms[tel_id].cam_rotation)
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
            moments = None
            try:
                moments_cam = hillas_parameters(event.inst.pixel_pos[tel_id][0],
                                            event.inst.pixel_pos[tel_id][1],
                                            pmt_signal*mask)

                moments = hillas_parameters(nom_coord.x, nom_coord.y,pmt_signal*mask)

            except HillasParameterizationError as e:
                print(e)
                continue

            # Make cut based on Hillas parameters
            if self.preselect(moments, tel_id):

                # Dialte around edges of image
                for i in range(5):
                    dilate(self.geoms[tel_id], mask)

                # Save everything in dicts for reconstruction later
                pixel_area[tel_id] = self.geoms[tel_id].pix_area

                pixel_x[tel_id] = nom_coord.x[mask]
                pixel_y[tel_id] = nom_coord.y[mask]

                tel_x[tel_id] = tilt_tel.x
                tel_y[tel_id] = tilt_tel.y

                tel_type[tel_id] = self.geoms[tel_id].cam_id
                image[tel_id] = pmt_signal[mask]

                hillas[tel_id] = moments_cam
                hillas_nom[tel_id] = moments

                tel_phi[tel_id] = array_pointing.az
                tel_theta[tel_id] = 90 * u.deg - array_pointing.alt

        # Cut on number of telescopes remaining
        if len(image)>1:
            #self.viewer.draw_event(event, hillas_nom)
            #plt.show()

            # Perform Hillas based reconstruction
            fit_result = self.fit.predict(hillas, event.inst, tel_phi, tel_theta)

            fit_result.core_x = scipy.random.normal(event.mc.core_x, 25 * u.m) * u.m
            fit_result.core_y = scipy.random.normal(event.mc.core_y, 25 * u.m) * u.m

            print(fit_result)

            energy_result = ReconstructedEnergyContainer()
            energy_result.energy = scipy.random.normal(event.mc.energy,
                                                       event.mc.energy * 0.2) * u.TeV

            # Perform ImPACT reconstruction
            self.ImPACT.set_event_properties(image, pixel_x, pixel_y, pixel_area,
                                             tel_type, tel_x, tel_y, array_pointing)
            ImPACT_shower, ImPACT_energy = self.ImPACT.predict(fit_result, energy_result)

            # insert the row into the table
            self.output.add_row((event.dl0.event_id, ImPACT_shower.alt,
                                 ImPACT_shower.az, ImPACT_energy.energy,
                                 event.mc.alt, event.mc.az, event.mc.energy))
def main():
    exe = ImPACTReconstruction()
    exe.run()

if __name__ == '__main__':
    main()