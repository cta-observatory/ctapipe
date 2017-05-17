from ctapipe.utils.datasets import get_example_simtelarray_file
from ctapipe.io.hessio import hessio_event_source
from ctapipe.core import Container
from ctapipe.coordinates import GroundFrame, TiltedGroundFrame

from ctapipe.io.containers import RawData
from ctapipe.io.containers import MCShowerData, CentralTriggerData
from ctapipe.image.hillas import hillas_parameters
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe import io
from astropy.coordinates import Angle, AltAz
from astropy.time import Time
from ctapipe.instrument import InstrumentDescription as ID
from ctapipe.coordinates import CameraFrame, NominalFrame

from astropy import units as u

import numpy as np
import pyhessio
import time

import logging
import argparse
logging.basicConfig(level=logging.DEBUG)


def get_mc_calibration_coeffs(tel_id):
    """
    Get the calibration coefficients from the MC data file to the
    data.  This is ahack (until we have a real data structure for the
    calibrated data), it should move into `ctapipe.io.hessio_event_source`.

    returns
    -------
    (peds,gains) : arrays of the pedestal and pe/dc ratios.
    """
    peds = pyhessio.get_pedestal(tel_id)[0]
    gains = pyhessio.get_calibration(tel_id)[0]
    return peds, gains


def apply_mc_calibration(adcs, tel_id):
    """
    apply basic calibration
    """
    peds, gains = get_mc_calibration_coeffs(tel_id)
    #print(adcs,peds,gains)
    #print ((adcs - peds) * gains)
    return (adcs - peds) * gains


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Perform simple Hillas Reco')
    parser.add_argument('filename', metavar='EVENTIO_FILE', nargs='?',
                        default=get_example_simtelarray_file())
    args = parser.parse_args()

    source = hessio_event_source(args.filename)

    container = Container("hessio_container")
    container.meta.add_item('pixel_pos', dict())
    container.add_item("dl0", RawData())
    container.add_item("mc", MCShowerData())
    container.add_item("trig", CentralTriggerData())
    container.add_item("count")
    tel,cam,opt = ID.load(filename=args.filename)

    for event in source:
        start_t = time.time()

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

        print('Scanning input file... count = {}'.format(event.count))
        width = list()
        length = list()
        size = list()
        impact = list()

        table = "CameraTable_VersionFeb2016_TelID"
        grd_core_true = GroundFrame(x=np.asarray(container.mc.core_x) * u.m, y=np.asarray(container.mc.core_y) * u.m,
                                    z=np.asarray(0) * u.m)
        tilt_core_true = grd_core_true.transform_to(
            TiltedGroundFrame(pointing_direction=[container.mc.alt, container.mc.az]))

        for tel_id in container.dl0.tels_with_data:
            x, y = event.meta.pixel_pos[tel_id]

            geom = io.CameraGeometry.guess(x, y, event.meta.optical_foclen[tel_id])
            image = apply_mc_calibration(event.dl0.tel[tel_id].adc_sums[0], tel_id)
            clean_mask = tailcuts_clean(geom, image, 1, picture_thresh=15, boundary_thresh=8)

            camera_coord = CameraFrame(x=x, y=y, z=np.zeros(x.shape)*u.m,
                                       focal_length=tel['TelescopeTable_VersionFeb2016'][tel['TelescopeTable_VersionFeb2016']['TelID']==tel_id]['FL'][0]*u.m)

            nom_coord = camera_coord.transform_to(NominalFrame(array_direction=[container.mc.alt,container.mc.az],
                                                               pointing_direction=[container.mc.alt,container.mc.az],
                                                               ))
            tx = tel['TelescopeTable_VersionFeb2016'][tel['TelescopeTable_VersionFeb2016']['TelID'] == tel_id]['TelX'][0]
            ty = tel['TelescopeTable_VersionFeb2016'][tel['TelescopeTable_VersionFeb2016']['TelID'] == tel_id]['TelY'][0]
            tz = tel['TelescopeTable_VersionFeb2016'][tel['TelescopeTable_VersionFeb2016']['TelID'] == tel_id]['TelZ'][0]

            grd_tel = GroundFrame(x=tx*u.m, y=ty*u.m, z=tz*u.m)
            tilt_tel = grd_tel.transform_to(TiltedGroundFrame(pointing_direction=[container.mc.alt, container.mc.az]))
            nom_x = nom_coord.x
            nom_y = nom_coord.y

            if np.sum(image*clean_mask) >40:
                hill = hillas_parameters(nom_x,nom_y,image*clean_mask)
                print(hill)
                if hill[0].size > 100:
                    if tel_id<500:
                        print(width,length,tilt_tel)

                        width.append(hill[0].width)
                        length.append(hill[0].length)
                        size.append(hill[0].size)
                        impact = tilt_core_true.separation_3d(tilt_tel)



