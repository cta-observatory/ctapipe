from ctapipe.utils.datasets import get_example_simtelarray_file
from ctapipe.io.hessio import hessio_event_source
from ctapipe.core import Container
from ctapipe.coordinates.frames import GroundFrame, TiltedGroundFrame, project_to_ground

from ctapipe.io.containers import RawData
from ctapipe.io.containers import MCShowerData, CentralTriggerData
from ctapipe.reco.hillas import hillas_parameters
from ctapipe.reco.weighted_axis_minimisation import WeightedAxisMinimisation
from ctapipe.reco.cleaning import tailcuts_clean
from ctapipe import io
from astropy.coordinates import Angle, AltAz
from astropy.time import Time
from ctapipe.instrument import InstrumentDescription as ID
from ctapipe.coordinates import CameraFrame, NominalFrame, TelescopeFrame

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
        #print(event.trig)
        #print(event.mc)
        #print(event.dl0)

        hillas_parameter_list = list()
        tel_x = list()
        tel_y = list()
        tel_z = list()
        pix_x = list()
        pix_y = list()
        pix_weight = list()

        table = "CameraTable_VersionFeb2016_TelID"
        for tel_id in container.dl0.tels_with_data:

            #x, y = cam[table+str(tel_id)]['PixX'][:],cam[table+str(tel_id)]['PixY'][:]
            x, y = event.meta.pixel_pos[tel_id]

            start_t = time.time()
            #geom = cam[table+str(tel_id)]['PixNeig'][:]
            #print(event.meta.optical_foclen[tel_id])
            geom = io.CameraGeometry.guess(x, y,event.meta.optical_foclen[tel_id])
            image = apply_mc_calibration(event.dl0.tel[tel_id].adc_sums[0], tel_id)

            clean_mask = tailcuts_clean(geom,image,1,picture_thresh=12,boundary_thresh=6)

            camera_coord = CameraFrame(x=x,y=y,z=np.zeros(x.shape)*u.m)

            nom_coord = camera_coord.transform_to(NominalFrame(array_direction=[container.mc.alt,container.mc.az],
                                                       pointing_direction=[container.mc.alt,container.mc.az],
                                                       focal_length=tel['TelescopeTable_VersionFeb2016'][tel['TelescopeTable_VersionFeb2016']['TelID']==tel_id]['FL'][0]*u.m))
            nom_x = nom_coord.x
            nom_y = nom_coord.y

            hill = hillas_parameters(nom_x,nom_y,image*clean_mask)

            if hill.size > 100:
                hillas_parameter_list.append(hill)
                tel_x.append(tel['TelescopeTable_VersionFeb2016'][tel['TelescopeTable_VersionFeb2016']['TelID']==tel_id]['TelX'][0])
                tel_y.append(tel['TelescopeTable_VersionFeb2016'][tel['TelescopeTable_VersionFeb2016']['TelID']==tel_id]['TelY'][0])
                tel_z.append(tel['TelescopeTable_VersionFeb2016'][tel['TelescopeTable_VersionFeb2016']['TelID']==tel_id]['TelZ'][0])
                pix_x.append(nom_x.to(u.deg).value)
                pix_y.append(nom_y.to(u.deg).value)
                pix_weight.append(image)

        grd_core_true = GroundFrame(x=np.asarray(container.mc.core_x)*u.m, y=np.asarray(container.mc.core_y)*u.m, z=np.asarray(0)*u.m)
        tilt_core_true = grd_core_true.transform_to(TiltedGroundFrame(pointing_direction=[container.mc.alt,container.mc.az]))

        print("Hillas time",time.time()-start_t)
        if len(hillas_parameter_list)>1:
            print (len(hillas_parameter_list))
            #start_t = time.time()
            grd = GroundFrame(x=np.asarray(tel_x)*u.m, y=np.asarray(tel_y)*u.m, z=np.asarray(tel_z)*u.m)
            tilt = grd.transform_to(TiltedGroundFrame(pointing_direction=[container.mc.alt,container.mc.az]))
            print(tilt.x)

            reco = WeightedAxisMinimisation()
            shower_direction = reco.reconstruct_event(hillas_parameter_list,tilt.x.value.tolist(),tilt.y.value.tolist(),
                                                      pix_x,pix_y,pixel_weight=pix_weight,
                                                      shower_seed=[tilt_core_true.x.value,tilt_core_true.y.value])

            print(shower_direction)

            print(container.mc.alt*57.3,container.mc.az*57.3)
            print(tilt_core_true.x,tilt_core_true.y)
        print("reconstruction time",time.time()-start_t)