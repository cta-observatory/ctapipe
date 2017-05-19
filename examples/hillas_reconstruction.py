from ctapipe.utils.datasets import get_example_simtelarray_file
from ctapipe.io.hessio import hessio_event_source
from ctapipe.core import Container
from ctapipe.coordinates.frames import GroundFrame, TiltedGroundFrame, project_to_ground

from ctapipe.io.containers import RawData
from ctapipe.io.containers import MCShowerData, CentralTriggerData
from ctapipe.reco.hillas import hillas_parameters
import ctapipe.reco.hillas_intersection as hill_int
from ctapipe.reco.cleaning import tailcuts_clean
from ctapipe import io
from astropy.coordinates import Angle, AltAz
from astropy.time import Time
from ctapipe.instrument import InstrumentDescription as ID
from ctapipe.coordinates import CameraFrame, NominalFrame
from astropy import units as u
from astropy.units import Quantity

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

        hillas_parameter_list = list()
        tel_x = list()
        tel_y = list()
        tel_z = list()

        table = "CameraTable_VersionFeb2016_TelID"
        for tel_id in container.dl0.tels_with_data:

            x, y = cam[table+str(tel_id)]['PixX'][:],cam[table+str(tel_id)]['PixY'][:]
            start_t = time.time()
            geom = cam[table+str(tel_id)]['PixNeig'][:]
            print(event.dl0.tel[tel_id].adc_sums[0],event.dl0.tel[tel_id].adc_sums[1])
            image = apply_mc_calibration(event.dl0.tel[tel_id].adc_sums[0], tel_id)

            x = Quantity(np.asanyarray(x, dtype=np.float64)).value
            y = Quantity(np.asanyarray(y, dtype=np.float64)).value

            camera_coord = CameraFrame(x=x*u.m,y=y*u.m,z=np.zeros(x.shape)*u.m)
            nom_coord = camera_coord.transform_to(NominalFrame(array_direction=[container.mc.alt,container.mc.az],
                                                       pointing_direction=[container.mc.alt,container.mc.az],
                                                       focal_length=tel['TelescopeTable_VersionFeb2016'][tel['TelescopeTable_VersionFeb2016']['TelID']==tel_id]['FL'][0]*u.m))
                                                               #,
                                                       #rotation=0*u.deg))
            nom_x = nom_coord.x
            nom_y = nom_coord.y

            clean_mask = tailcuts_clean(cam[table+str(tel_id)],image,1,picture_thresh=12,boundary_thresh=6)
            hill = hillas_parameters(nom_x,nom_y,image*clean_mask)

            if hill.size > 100:
                hillas_parameter_list.append(hill)
                tel_x.append(tel['TelescopeTable_VersionFeb2016'][tel['TelescopeTable_VersionFeb2016']['TelID']==tel_id]['TelX'][0])
                tel_y.append(tel['TelescopeTable_VersionFeb2016'][tel['TelescopeTable_VersionFeb2016']['TelID']==tel_id]['TelY'][0])
                tel_z.append(tel['TelescopeTable_VersionFeb2016'][tel['TelescopeTable_VersionFeb2016']['TelID']==tel_id]['TelZ'][0])

                #tel_config_list.append(tel[tel_id])
        print("Hillas time",time.time()-start_t)
        if len(hillas_parameter_list)>1:
            #start_t = time.time()
            grd = GroundFrame(x=np.asarray(tel_x)*u.m, y=np.asarray(tel_y)*u.m, z=np.asarray(tel_z)*u.m)
            tilt = grd.transform_to(TiltedGroundFrame(pointing_direction=[70*u.deg,0*u.deg]))

            shower_direction = hill_int.reconstruct_nominal(hillas_parameter_list)
            core_position = hill_int.reconstruct_tilted(hillas_parameter_list,tilt.x.value.tolist(),tilt.y.value.tolist())
            #print(project_to_ground(core_position))

            print(shower_direction)
            print(container.mc.core_x,container.mc.core_y)
            print(core_position)
        print("reconstruction time",time.time()-start_t)