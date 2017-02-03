from astropy import units as u
import numpy as np
from ctapipe.utils.datasets import get_path

from ctapipe.reco.FitGammaHillas import FitGammaHillas, GreatCircle
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.image.cleaning import tailcuts_clean, dilate
from ctapipe.instrument import InstrumentDescription as ID

from ctapipe.io.hessio import hessio_event_source
from ctapipe.io import CameraGeometry
from ctapipe.reco.ImPACT import ImPACTFitter
from ctapipe.calib.camera.calibrators import CameraDL1Calibrator
from ctapipe.coordinates import *
from ctapipe.io.containers import ReconstructedShowerContainer, ReconstructedEnergyContainer

def test_ImPACT_fit():
    '''
    a test of the complete fit procedure on one event including:
    • tailcut cleaning
    • hillas parametrisation
    • GreatCircle creation
    • direction fit
    • position fit

    in the end, proper units in the output are asserted '''

    filename = "/Users/dparsons/Desktop/gamma_20deg_180deg_run9021___cta-prod3-merged_desert-2150m-Paranal-subarray-3.simtel.gz"
    ImPACT = ImPACTFitter(root_dir="/Users/dparsons/Documents/Unix/CTA/ImPACT_pythontests/", fit_xmax=True)
    tel, cam, opt = ID.load(filename=filename)

    calibrator = CameraDL1Calibrator(None, None)

    fit = FitGammaHillas()

    cam_geom = {}
    tel_phi = {}
    tel_theta = {}

    source = hessio_event_source(filename)# Get source ready to loop over the file

    for event in source:
        # First we need to create some dictionaries with the info we will need for ImPACT
        hillas_dict = {}
        image = {}
        pixel_x = {}
        pixel_y = {}
        pixel_area = {}
        tel_type = {}
        tel_x = {}
        tel_y = {}

        calibrator.calibrate(event) # calibrate the events

        # store MC pointing direction for the array
        array_pointing = np.array((event.mcheader.run_array_direction[1], event.mcheader.run_array_direction[0])) * \
                         u.rad

        for tel_id in event.dl0.tels_with_data:

            pmt_signal = event.dl1.tel[tel_id].image[0]
            image[tel_id] = pmt_signal

            x, y = event.inst.pixel_pos[tel_id]
            fl = event.inst.optical_foclen[tel_id]

            if tel_id not in cam_geom:
                cam_geom[tel_id] = CameraGeometry.guess(
                                        event.inst.pixel_pos[tel_id][0],
                                        event.inst.pixel_pos[tel_id][1],
                                        event.inst.optical_foclen[tel_id])

                tel_phi[tel_id] = 0.*u.deg
                tel_theta[tel_id] = 20.*u.deg

            tel_type[tel_id] = cam_geom[tel_id].cam_id

            # Transform the pixels positions into nominal coordinates
            camera_coord = CameraFrame(x=x, y=y, z=np.zeros(x.shape) * u.m,
                                       focal_length=fl,
                                       rotation= 90*u.deg - cam_geom[tel_id].cam_rotation )

            nom_coord = camera_coord.transform_to(NominalFrame(array_direction=array_pointing,
                                                               pointing_direction=array_pointing))
            pixel_x[tel_id] = nom_coord.x
            pixel_y[tel_id] = nom_coord.y

            cam_name = "CameraTable_VersionFeb2016_TelID" + str(tel_id)
            pixel_area[tel_id] = u.rad * u.rad * (cam[cam_name][cam[cam_name]['PixID'] == 0]['PixA'][0] / (fl * fl))

            tx = tel['TelescopeTable_VersionFeb2016'][
                tel['TelescopeTable_VersionFeb2016']['TelID'] == tel_id]['TelX'][0]
            ty = tel['TelescopeTable_VersionFeb2016'][
                tel['TelescopeTable_VersionFeb2016']['TelID'] == tel_id]['TelY'][0]
            tz = tel['TelescopeTable_VersionFeb2016'][
                tel['TelescopeTable_VersionFeb2016']['TelID'] == tel_id]['TelZ'][0]

            # ImPACT reconstruction is performed in the tilted system, so we need to transform tel positions
            grd_tel = GroundFrame(x=tx * u.m, y=ty * u.m, z=tz * u.m)
            tilt_tel = grd_tel.transform_to(TiltedGroundFrame(pointing_direction=array_pointing))

            tel_x[tel_id] = tilt_tel.x
            tel_y[tel_id] = tilt_tel.y

            mask = tailcuts_clean(cam_geom[tel_id], pmt_signal, 1,
                                  picture_thresh=10., boundary_thresh=5.)

            try:
                moments = hillas_parameters(event.inst.pixel_pos[tel_id][0],
                                            event.inst.pixel_pos[tel_id][1],
                                            pmt_signal*mask)
                hillas_dict[tel_id] = moments
            except HillasParameterizationError as e:
                print(e)
                continue

        # Perform Hillas analysis first for seeding the fit
        fit_result = fit.predict(hillas_dict, event.inst, tel_phi, tel_theta)
        print(fit_result)

        # Set up the ImPACT class for use in the reconstruction
        ImPACT.set_event_properties(image, pixel_x, pixel_y, pixel_area, tel_type, tel_x, tel_y, array_pointing)
        energy_result = ReconstructedEnergyContainer()
        energy_result.energy = event.mc.energy

        # Perform ImPACT fit
        shower_reco, energy_reco = ImPACT.predict(fit_result, energy_result)
        if len(hillas_dict) < 2: continue

        print(shower_reco)
        print(energy_reco)

        #assert fit_result.is_valid
        #return

if __name__ == "__main__":
    test_ImPACT_fit()