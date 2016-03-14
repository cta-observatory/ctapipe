from ctapipe.utils.datasets import get_example_simtelarray_file
from ctapipe.io.hessio import hessio_event_source
from ctapipe.core import Container

from ctapipe.io.containers import RawData
from ctapipe.io.containers import RawCameraData, MCShowerData, CentralTriggerData
from ctapipe.reco.hillas import hillas_parameters
from ctapipe.reco.cleaning import tailcuts_clean
from ctapipe import visualization, io
from astropy.coordinates import Angle
from astropy.time import Time

from matplotlib import pyplot as plt
from astropy import units as u
import pyhessio

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
        print(event.trig)
        print(event.mc)
        print(event.dl0)

        hillas_parameter_list = list()

        for tel_id in container.dl0.tels_with_data:
            x, y = event.meta.pixel_pos[tel_id]
            geom = io.CameraGeometry.guess(x, y)
            image = apply_mc_calibration(event.dl0.tel[tel_id].adc_sums[0], tel_id)

            clean_mask = tailcuts_clean(geom,image,1,picture_thresh=10,boundary_thresh=5)
            hill = hillas_parameters(x,y,image*clean_mask)

            if hill.size > 100:
                hillas_parameter_list.append(hill)

        print (len(hillas_parameter_list))

