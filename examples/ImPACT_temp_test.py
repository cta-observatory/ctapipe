from ctapipe.utils.datasets import get_example_simtelarray_file
from ctapipe.io.hessio import hessio_event_source
from ctapipe.core import Container

from ctapipe.io.containers import RawData
from ctapipe.io.containers import MCShowerData, CentralTriggerData
from ctapipe.image.cleaning import tailcuts_clean, dilate

from ctapipe import io, visualization
from astropy.coordinates import Angle, AltAz
from astropy.time import Time
from ctapipe.instrument import InstrumentDescription as ID
from ctapipe.coordinates import CameraFrame, NominalFrame, GroundFrame, TiltedGroundFrame, HorizonFrame
from ctapipe.calib.camera.calibrators import calibration_parameters, calibrate_event

from astropy import units as u
import matplotlib.pyplot as plt

import numpy as np
import pyhessio

import logging
import argparse

logging.basicConfig(level=logging.DEBUG)

HB4 = [279, 280, 281, 282, 283, 284, 286, 287, 289, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 315,
       316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337,
       338, 345, 346, 347, 348, 349, 350, 375, 376, 377, 378, 379, 380, 393, 400, 402, 403, 404, 405, 406, 408, 410,
       411, 412, 413, 414, 415, 416, 417]

amp_cut = {"LSTCam": 70, "NectarCam": 65, "GATE": 30}


def rotate_translate(pixel_pos_x, pixel_pos_y, x_trans, y_trans, phi):
    """
    Function to perform rotation and translation of pixel lists
    Parameters
    ----------
    pixel_pos_x: ndarray
        Array of pixel x positions
    pixel_pos_y: ndarray
        Array of pixel x positions
    x_trans: float
        Translation of position in x coordinates
    y_trans: float
        Translation of position in y coordinates
    phi: float
        Rotation angle of pixels
    Returns
    -------
        ndarray,ndarray: Transformed pixel x and y coordinates
    """

    pixel_pos_trans_x = (pixel_pos_x - x_trans) * np.cos(phi) - (pixel_pos_y - y_trans) * np.sin(phi)
    pixel_pos_trans_y = (pixel_pos_x - x_trans) * np.sin(phi) + (pixel_pos_y - y_trans) * np.cos(phi)
    return pixel_pos_trans_x, pixel_pos_trans_y

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
    geom_dict = dict()
    sum = 0

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
        params = dict()
        params['integrator'] = 'nb_peak_integration'
        params['integration_window'] = (10,4)
        cal = calibrate_event(event, params, geom_dict)

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

            camera_coord = CameraFrame(x=x, y=y, z=np.zeros(x.shape) * u.m,
                                       focal_length=tel['TelescopeTable_VersionFeb2016'][
                                                        tel['TelescopeTable_VersionFeb2016']['TelID'] == tel_id]
                                                    ['FL'][0] * u.m,rotation=90*u.deg - geom.cam_rotation)

            nom_coord = camera_coord.transform_to(NominalFrame(array_direction=[container.mc.alt, container.mc.az],
                                                               pointing_direction=[container.mc.alt, container.mc.az]))

            x = nom_coord.x.to(u.deg)
            y = nom_coord.y.to(u.deg)

            tx = tel['TelescopeTable_VersionFeb2016'][
                 tel['TelescopeTable_VersionFeb2016']['TelID'] == tel_id]['TelX'][0]
            ty = tel['TelescopeTable_VersionFeb2016'][
                 tel['TelescopeTable_VersionFeb2016']['TelID'] == tel_id]['TelY'][0]
            tz = tel['TelescopeTable_VersionFeb2016'][
                 tel['TelescopeTable_VersionFeb2016']['TelID'] == tel_id]['TelZ'][0]

            grd_tel = GroundFrame(x=tx * u.m, y=ty * u.m, z=tz * u.m)
            tilt_tel = grd_tel.transform_to(TiltedGroundFrame(pointing_direction=[container.mc.alt, container.mc.az]))

            grd_core_true = GroundFrame(x=np.asarray(container.mc.core_x)*u.m, y=np.asarray(container.mc.core_y)*u.m,
                                        z=np.asarray(0)*u.m)
            tilt_core_true = grd_core_true.transform_to(TiltedGroundFrame(
                pointing_direction=[container.mc.alt, container.mc.az]))
            phi = np.arctan2((tilt_tel.x - tilt_core_true.x), (tilt_tel.y - tilt_core_true.y))

            energy = container.mc.energy
            point = HorizonFrame(alt=container.mc.alt, az=container.mc.az)
            source = point.transform_to(NominalFrame(array_direction=[container.mc.alt, container.mc.az]))
            print(source)

            pix_x_rot, pix_y_rot = rotate_translate(x, y, source.x, source.y, phi)
            h = plt.hist2d(pix_x_rot, pix_y_rot, weights=image, bins=[150,150], range=[[-5,5],[-5,5]])

            if sum is not 0:
                sum += h[0]
            else:
                sum = h[0]
            ev+=1
            if ev > 100:
                break
    plt.close()
    plt.imshow(sum)
    plt.show()