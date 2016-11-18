from ctapipe.utils.datasets import get_example_simtelarray_file
from ctapipe.io.hessio import hessio_event_source
from ctapipe.core import Container

from ctapipe.io.containers import RawData
from ctapipe.io.containers import MCShowerData, CentralTriggerData
from ctapipe.image.cleaning import tailcuts_clean, dilate
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError

from ctapipe import io, visualization
from astropy.coordinates import Angle, AltAz
from astropy.time import Time
from ctapipe.instrument import InstrumentDescription as ID
from ctapipe.coordinates import CameraFrame, NominalFrame, GroundFrame, TiltedGroundFrame
from ctapipe.reco.ImPACT import ImPACTFitter
from ctapipe.calib.camera.calibrators import calibration_parameters, calibrate_event

from astropy import units as u

import numpy as np
import pyhessio
import matplotlib.pylab as plt

import logging
import argparse

logging.basicConfig(level=logging.DEBUG)

HB4 = [1, 2, 3, 71, 72, 73, 74, 75, 76, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 104, 105, 106, 107, 108, 109,
       279, 280, 281, 282, 283, 284, 286, 287, 289, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 315,
       316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337,
       338, 345, 346, 347, 348, 349, 350, 375, 376, 377, 378, 379, 380, 393, 400, 402, 403, 404, 405, 406, 408, 410,
       411, 412, 413, 414, 415, 416, 417]

amp_cut = {"LSTCam": 70, "NectarCam": 65, "GATE": 30}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform simple Hillas Reco')
    parser.add_argument('filename', metavar='EVENTIO_FILE', nargs='?',
                        default=get_example_simtelarray_file())
    args = parser.parse_args()

    source = hessio_event_source(args.filename,
                                 allowed_tels=HB4)

    container = Container("hessio_container")
    container.meta.add_item('pixel_pos', dict())
    container.add_item("dl0", RawData())
    container.add_item("mc", MCShowerData())
    container.add_item("trig", CentralTriggerData())
    container.add_item("count")
    tel, cam, opt = ID.load(filename=args.filename)
    ev = 0

    impact = list()
    geom = 0
    ImPACT = ImPACTFitter()

    geom_dict = dict()

    energy_list = list()
    reconstructed_energy_list = list()
    theta_sqr_list = list()

    for event in source:

        container.dl0.tels_with_data = set(pyhessio.get_teldata_list())

        container.trig.tels_with_trigger \
            = pyhessio.get_central_event_teltrg_list()
        time_s, time_ns = pyhessio.get_central_event_gps_time()
        container.trig.gps_time = Time(time_s * u.s, time_ns * u.ns,
                                       format='gps', scale='utc')
        container.mc.energy = pyhessio.get_mc_shower_energy() * u.TeV
        container.mc.alt = Angle(pyhessio.get_mc_shower_altitude(), u.rad)
        container.mc.az = Angle(pyhessio.get_mc_shower_azimuth(), u.rad)
        container.mc.core_x = pyhessio.get_mc_event_xcore() * u.m
        container.mc.core_y = pyhessio.get_mc_event_ycore() * u.m

        # this should be done in a nicer way to not re-allocate the
        # data each time (right now it's just deleted and garbage
        # collected)

        container.dl0.tel = dict()  # clear the previous telescopes

        table = "CameraTable_VersionFeb2016_TelID"
        pix_x = list()
        pix_y = list()
        pix_a = list()

        tel_x = list()
        tel_y = list()
        image_list = list()
        type_tel = list()
        tel_id_list = list()

        clean = list()
        geom = None
        params = dict()
        params['integrator'] = 'nb_peak_integration'
        params['integration_window'] = (10,4)
        cal = calibrate_event(event, params, geom_dict)
        #if container.mc.energy < 0.1*u.TeV:
        #    continue

        for tel_id in container.dl0.tels_with_data:

            used = np.any(HB4 == tel_id)

            if not used:
                continue

            x, y = event.meta.pixel_pos[tel_id]

            if geom_dict is not None and tel_id in geom_dict:
                geom = geom_dict[tel_id]
            else:
                geom = io.CameraGeometry.guess(event.meta.pixel_pos[tel_id][0], event.meta.pixel_pos[tel_id][1],
                                            event.meta.optical_foclen[tel_id])
                if geom_dict is not None:
                    geom_dict[tel_id] = geom

            image = cal.dl1.tel[tel_id].pe_charge

            fl = tel['TelescopeTable_VersionFeb2016'][
                    tel['TelescopeTable_VersionFeb2016']['TelID'] == tel_id]['FL'][0]

            clean_mask = tailcuts_clean(geom, image, 1, picture_thresh=7, boundary_thresh=14)
            camera_coord = CameraFrame(x=x, y=y, z=np.zeros(x.shape) * u.m,
                                       focal_length=tel['TelescopeTable_VersionFeb2016'][
                                                        tel['TelescopeTable_VersionFeb2016']['TelID'] == tel_id]
                                                    ['FL'][0] * u.m,
                                       rotation=90*u.deg - geom.cam_rotation + 180 * u.deg)

            nom_coord = camera_coord.transform_to(NominalFrame(array_direction=[container.mc.alt, container.mc.az],
                                                               pointing_direction=[container.mc.alt, container.mc.az]))

            x = nom_coord.x.to(u.deg)
            y = nom_coord.y.to(u.deg)

            try:
                hillas = hillas_parameters(x, y, image * clean_mask)[0]
            except HillasParameterizationError:
                continue

            dilate(geom, clean_mask)
            dilate(geom, clean_mask)
            #dilate(geom, clean_mask)

            if hillas.size > amp_cut[geom.cam_id]:
                pix_x.append(x[clean_mask])
                pix_y.append(y[clean_mask])
                image_list.append(image[clean_mask])
                #pix_x.append(x)
                #pix_y.append(y)
                #image_list.append(image)

                type_tel.append(geom.cam_id)
                tel_id_list.append(tel_id)
                clean.append(clean_mask)

                cam_name = "CameraTable_VersionFeb2016_TelID" + str(tel_id)
                area = u.rad*u.rad*(cam[cam_name][cam[cam_name]['PixID'] == 0]['PixA'][0]/(fl*fl))
                pix_a.append(area)

                tx = tel['TelescopeTable_VersionFeb2016'][
                    tel['TelescopeTable_VersionFeb2016']['TelID'] == tel_id]['TelX'][0]
                ty = tel['TelescopeTable_VersionFeb2016'][
                    tel['TelescopeTable_VersionFeb2016']['TelID'] == tel_id]['TelY'][0]
                tz = tel['TelescopeTable_VersionFeb2016'][
                    tel['TelescopeTable_VersionFeb2016']['TelID'] == tel_id]['TelZ'][0]

                grd_tel = GroundFrame(x=tx*u.m, y=ty*u.m, z=tz*u.m)
                tilt_tel = grd_tel.transform_to(TiltedGroundFrame(pointing_direction=[container.mc.alt, container.mc.az]))
                tel_x.append(tilt_tel.x)
                tel_y.append(tilt_tel.y)

        if len(tel_x)<2:
            continue
        grd_core_true = GroundFrame(x=np.asarray(container.mc.core_x)*u.m, y=np.asarray(container.mc.core_y)*u.m, z=np.asarray(0)*u.m)
        tilt_core_true = grd_core_true.transform_to(TiltedGroundFrame(pointing_direction=[container.mc.alt,container.mc.az]))

        energy = container.mc.energy

        ImPACT.set_event_properties(image_list, pix_x, pix_y, pix_a, type_tel, tel_x, tel_y)
        params = ImPACT.fit_event(np.random.normal(0,0.1)*u.deg, np.random.normal(0,0.1)*u.deg,
                                  np.random.normal(tilt_core_true.x.value, 20)*u.m,
                                  np.random.normal(tilt_core_true.y.value, 20)*u.m,
                                  np.random.normal(energy.value,energy.value*0.15)*u.TeV)

        print(tilt_core_true.x, tilt_core_true.y, energy)

        src_x = params["source_x"] * u.deg
        src_y = params["source_y"] * u.deg
        reconstructed_energy = params["energy"] * u.TeV
        print("ThetaSqr",src_x*src_x + src_y*src_y,"Energy Bias",reconstructed_energy/energy)
        energy_list.append(energy.value)
        reconstructed_energy_list.append(reconstructed_energy/energy)
        theta_sqr_list.append((src_x*src_x + src_y*src_y).value)
        ev += 1
        if ev > 1000:
            break
        draw = False
        if draw:
            for tel_num in range(len(tel_x)):
                fig, axs = plt.subplots(1, 3, figsize=(24, 8), sharey=True, sharex=True)

                prediction = ImPACT.get_prediction(tel_num, 90 * u.deg - container.mc.alt, container.mc.az * u.rad,
                                                   params["core_x"]*u.m,  params["core_y"]*u.m,  params["energy"]*u.TeV,
                                                   params["x_max_scale"])

                disp = visualization.CameraDisplay(geom_dict[tel_id_list[tel_num]], ax=axs[0], title="Image")
                disp.image = image_list[tel_num]
                disp.add_colorbar(ax=axs[0])

                disp_pred = visualization.CameraDisplay(geom_dict[tel_id_list[tel_num]], ax=axs[1], title="Prediction")
                disp_pred.image = prediction
                disp_pred.add_colorbar(ax=axs[1])

                disp_resid = visualization.CameraDisplay(geom_dict[tel_id_list[tel_num]], ax=axs[2], title="Prediction")
                disp_resid.image = image_list[tel_num]-prediction
                disp_resid.add_colorbar(ax=axs[2])

                plt.show()




