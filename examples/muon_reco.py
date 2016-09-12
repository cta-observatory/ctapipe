from ctapipe.utils.datasets import get_example_simtelarray_file
from ctapipe.io.hessio import hessio_event_source
from ctapipe.core import Container

from ctapipe.io.containers import RawData
from ctapipe.io.containers import MCShowerData, CentralTriggerData
from ctapipe.reco.cleaning import tailcuts_clean
from ctapipe import io
from astropy.coordinates import Angle, AltAz
from astropy.time import Time
from ctapipe.instrument import InstrumentDescription as ID
from ctapipe.coordinates import CameraFrame, NominalFrame
from ctapipe.calib.array.muon_ring_finder import chaudhuri_kundu_circle_fit
from ctapipe.calib.array.muon_integrator import *

from ctapipe import visualization
import matplotlib.pyplot as plt

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
        table = "CameraTable_VersionFeb2016_TelID"

        for tel_id in container.dl0.tels_with_data:

            x, y = event.meta.pixel_pos[tel_id]

            geom = io.CameraGeometry.guess(x, y,event.meta.optical_foclen[tel_id])
            image = apply_mc_calibration(event.dl0.tel[tel_id].adc_sums[0], tel_id)
            clean_mask = tailcuts_clean(geom,image,1,picture_thresh=8,boundary_thresh=16)

            camera_coord = CameraFrame(x=x,y=y,z=np.zeros(x.shape)*u.m)

            nom_coord = camera_coord.transform_to(NominalFrame(array_direction=[container.mc.alt,container.mc.az],
                                                       pointing_direction=[container.mc.alt,container.mc.az],
                                                       focal_length=tel['TelescopeTable_VersionFeb2016'][tel['TelescopeTable_VersionFeb2016']['TelID']==tel_id]['FL'][0]*u.m))

            img = image*clean_mask
            noise = 5
            weight = img / (img+noise)

            centre_x,centre_y,radius = chaudhuri_kundu_circle_fit(x,y,image*clean_mask)
            dist = np.sqrt(np.power(x-centre_x,2) + np.power(y-centre_y,2))
            centre_x,centre_y,radius = chaudhuri_kundu_circle_fit(x,y,image*clean_mask*(dist<radius*1.5))
            dist = np.sqrt(np.power(x-centre_x,2) + np.power(y-centre_y,2))
            centre_x,centre_y,radius = chaudhuri_kundu_circle_fit(x,y,image*clean_mask*(dist<radius*1.2))

            print (centre_x,centre_y,radius)
            rad = list()
            cx = list()
            cy = list()
            
            mc_x = container.mc.core_x
            mc_y = container.mc.core_y

            if(np.sum(clean_mask*(dist<radius*1.2))>15):

                polygon = [(-15,-4),(-15,4),(-10,12),(9,12),(13,9.5),(16,4),(16,-4),(13.,-9.5),(9,-12),(-10,-12)]
                hole = [(2.48636, 2.376),(2.48636, -2.376),(-2.48636, -2.376),(-2.48636, 2.376)]
                hess = MuonLineIntegrate(polygon,hole,pixel_width=0.06*u.m)

                disp = visualization.CameraDisplay(geom)
                disp.image = image
                disp.add_ellipse((centre_x.value,centre_y.value),radius.value*2,radius.value*2,0,color="red")
                disp.cmap = "viridis"
                disp.add_colorbar()

                #hess.image_prediction(mc_x,mc_y,centre_x,centre_y,radius,0.1*u.m,x,y)
                if (image.shape[0]<2000):
                    hess.fit_muon(centre_x,centre_y,radius,x,y,image,mc_x.value,mc_y.value)

                plt.show()