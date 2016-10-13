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
from ctapipe.image.muon.muon_ring_finder import ChaudhuriKunduRingFitter
from ctapipe.image.muon.muon_integrator import *

from ctapipe import visualization
import matplotlib.pyplot as plt

from astropy import units as u
from IPython import embed

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
    #embed()
    ev = 0
    efficiency = list()
    efficiency.append(list())
    efficiency.append(list())
    efficiency.append(list())
    efficiency.append(list())


    impact = list()
    geom = 0
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

        for tel_id in container.dl0.tels_with_data:

            x, y = event.meta.pixel_pos[tel_id]
            if geom == 0:
                geom = io.CameraGeometry.guess(x, y,event.meta.optical_foclen[tel_id])
            image = apply_mc_calibration(event.dl0.tel[tel_id].adc_sums[0], tel_id)
            if image.shape[0] >1000:
                continue
            clean_mask = tailcuts_clean(geom,image,1,picture_thresh=5,boundary_thresh=7)

            camera_coord = CameraFrame(x=x,y=y,z=np.zeros(x.shape)*u.m)

            nom_coord = camera_coord.transform_to(NominalFrame(array_direction=[container.mc.alt,container.mc.az],
                                                       pointing_direction=[container.mc.alt,container.mc.az],
                                                       focal_length=tel['TelescopeTable_VersionFeb2016'][tel['TelescopeTable_VersionFeb2016']['TelID']==tel_id]['FL'][0]*u.m))

            x = nom_coord.x.to(u.deg)
            y = nom_coord.y.to(u.deg)

            img = image*clean_mask
            noise = 5
            weight = img / (img+noise)

            circlefitter = ChaudhuriKunduRingFitter()

            centre_x,centre_y,radius = circlefitter.fit(x,y,image*clean_mask)
            dist = np.sqrt(np.power(x-centre_x,2) + np.power(y-centre_y,2))
            ring_dist = np.abs(dist-radius)

            centre_x,centre_y,radius = circlefitter.fit(x,y,image*(ring_dist<radius*0.3))

            dist = np.sqrt(np.power(x-centre_x,2) + np.power(y-centre_y,2))
            ring_dist = np.abs(dist-radius)

            centre_x,centre_y,radius = circlefitter.fit(x,y,image*(ring_dist<radius*0.3))

            dist_mask = np.abs(dist-radius)<radius*0.4

            #print (centre_x,centre_y,radius)
            rad = list()
            cx = list()
            cy = list()
            
            mc_x = container.mc.core_x
            mc_y = container.mc.core_y
            pix_im = image*dist_mask
            nom_dist = np.sqrt(np.power(centre_x,2)+np.power(centre_y,2))
            if(np.sum(pix_im>5)>30 and np.sum(pix_im)>80 and nom_dist.value <1. and radius.value<1.5 and radius.value>1.):

                hess = MuonLineIntegrate(6.50431*u.m,0.883*u.m,pixel_width=0.16*u.deg)
                #telfit = MuonLineIntegrate()

                if (image.shape[0]<2000):
                    im,phi,width,eff=hess.fit_muon(centre_x,centre_y,radius,x[dist_mask],y[dist_mask],image[dist_mask])
                    #im,phi,width,eff=telfit.fit_muon(centre_x,centre_y,radius,x[dist_mask],y[dist_mask],image[dist_mask])
                    if( im < 6*u.m and im>0.9*u.m and width<0.08*u.deg and width>0.04*u.deg ):# and radius.value>0.2 and radius.value<0.4):
                        efficiency[tel_id-1].append(eff)
                        impact.append(im)

                    #print(len(efficiency),len(impact))
        ev +=1

    print("Muon Efficiency of CT1",np.average(np.asarray(efficiency[0])))
    print("Muon Efficiency of CT2",np.average(np.asarray(efficiency[1])))
    print("Muon Efficiency of CT3",np.average(np.asarray(efficiency[2])))
    print("Muon Efficiency of CT4",np.average(np.asarray(efficiency[3])))

    fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharey=False, sharex=False)

    axs[0][0].hist((efficiency[0]),bins=40,range=(0,0.1), alpha=0.5)
    axs[0][1].hist((efficiency[1]),bins=40,range=(0,0.1), alpha=0.5)
    axs[1][0].hist((efficiency[2]),bins=40,range=(0,0.1), alpha=0.5)
    axs[1][1].hist((efficiency[3]),bins=40,range=(0,0.1), alpha=0.5)

    plt.show()
