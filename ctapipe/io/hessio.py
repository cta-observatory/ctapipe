# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read HESSIO data.  

This requires the hessio python library to be installed
"""
import logging

from .containers import DataContainer

from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time

logger = logging.getLogger(__name__)

try:
    from pyhessio import open_hessio
    from pyhessio import HessioError
    from pyhessio import HessioTelescopeIndexError
    from pyhessio import HessioGeneralError
except ImportError as err:
    logger.fatal("the `pyhessio` python module is required to access MC data: {}"
                 .format(err))
    raise err

__all__ = [
    'hessio_event_source',
]


def hessio_event_source(url, max_events=None, allowed_tels=None,
                        requested_event=None, use_event_id=False):
    """A generator that streams data from an EventIO/HESSIO MC data file
    (e.g. a standard CTA data file.)

    Parameters
    ----------
    url : str
        path to file to open
    max_events : int, optional
        maximum number of events to read
    allowed_tels : list[int]
        select only a subset of telescope, if None, all are read. This can
        be used for example emulate the final CTA data format, where there
        would be 1 telescope per file (whereas in current monte-carlo,
        they are all interleaved into one file)
    requested_event : int
        Seek to a paricular event index
    use_event_id : bool
        If True ,'requested_event' now seeks for a particular event id instead
        of index
    """
    try:
        with open_hessio(url) as pyhessio:
        # the container is initialized once, and data is replaced within
        # it after each yield
            counter = 0
            eventstream = pyhessio.move_to_next_event()
            if allowed_tels is not None:
                allowed_tels = set(allowed_tels)
            data = DataContainer()
            data.meta['source'] = "hessio"

            # some hessio_event_source specific parameters
            data.meta['hessio__input'] =  url
            data.meta['hessio__max_events'] = max_events

            for event_id in eventstream:

                # Seek to requested event
                if requested_event is not None:
                    current = counter
                    if use_event_id:
                        current = event_id
                    if not current == requested_event:
                        counter += 1
                        continue

                data.dl0.run_id = pyhessio.get_run_number()
                data.dl0.event_id = event_id
                data.dl0.tels_with_data = set(pyhessio.get_teldata_list())

                # handle telescope filtering by taking the intersection of
                # tels_with_data and allowed_tels
                if allowed_tels is not None:
                    selected = data.dl0.tels_with_data & allowed_tels
                    if len(selected) == 0:
                        continue  # skip event
                    data.dl0.tels_with_data = selected

                data.trig.tels_with_trigger \
                    = pyhessio.get_central_event_teltrg_list()
                time_s, time_ns = pyhessio.get_central_event_gps_time()
                data.trig.gps_time = Time(time_s * u.s, time_ns * u.ns,
                                           format='gps', scale='utc')
                data.mc.energy = pyhessio.get_mc_shower_energy() * u.TeV
                data.mc.alt = Angle(pyhessio.get_mc_shower_altitude(), u.rad)
                data.mc.az = Angle(pyhessio.get_mc_shower_azimuth(), u.rad)
                data.mc.core_x = pyhessio.get_mc_event_xcore() * u.m
                data.mc.core_y = pyhessio.get_mc_event_ycore() * u.m
                data.mc.h_first_int = pyhessio.get_mc_shower_h_first_int() * u.m

                # mc run header data
                data.mcheader.run_array_direction = \
                    pyhessio.get_mc_run_array_direction()

                data.count = counter

                # this should be done in a nicer way to not re-allocate the
                # data each time (right now it's just deleted and garbage
                # collected)

                data.dl0.tel.clear()
                data.mc.tel.clear()  # clear the previous telescopes

                _fill_instrument_info(data,pyhessio)

                for tel_id in data.dl0.tels_with_data:

                    # event.mc.tel[tel_id] = MCCameraContainer()

                    data.mc.tel[tel_id].dc_to_pe \
                        = pyhessio.get_calibration(tel_id)
                    data.mc.tel[tel_id].pedestal \
                        = pyhessio.get_pedestal(tel_id)

                    # load the data per telescope/chan
                    # TODO: make this an array dim rather than dict
                    for chan in range(data.inst.num_channels[tel_id]):
                        data.dl0.tel[tel_id].adc_samples[chan] \
                            = pyhessio.get_adc_sample(channel=chan,
                                                      telescope_id=tel_id)
                        data.dl0.tel[tel_id].adc_sums[chan] \
                            = pyhessio.get_adc_sum(channel=chan,
                                                   telescope_id=tel_id)
                        data.mc.tel[tel_id].reference_pulse_shape[chan] = \
                            pyhessio.get_ref_shapes(tel_id, chan)

                    # load the data per telescope/pixel
                    data.mc.tel[tel_id].photo_electron_image \
                        = pyhessio.get_mc_number_photon_electron(telescope_id=tel_id)
                    data.mc.tel[tel_id].meta['refstep'] = pyhessio.get_ref_step(tel_id)
                    data.mc.tel[tel_id].time_slice = \
                        pyhessio.get_time_slice(tel_id)
                    data.mc.tel[tel_id].azimuth_raw = \
                        pyhessio.get_azimuth_raw(tel_id)
                    data.mc.tel[tel_id].altitude_raw = \
                        pyhessio.get_altitude_raw(tel_id)
                    data.mc.tel[tel_id].azimuth_cor = \
                        pyhessio.get_azimuth_cor(tel_id)
                    data.mc.tel[tel_id].altitude_cor = \
                        pyhessio.get_altitude_cor(tel_id)
                yield data
                counter += 1

                if max_events is not None and counter >= max_events:
                    pyhessio.close_file()
                    return
    except HessioError:
        raise RuntimeError("hessio_event_source failed to open '{}'"
                           .format(url))


def _fill_instrument_info(data, pyhessio, max_tel_id=1000):
    """
    fill the data.inst structure with instrumental information.

    Parameters
    ----------
    data: DataContainer
        data container to fill in

    """
    for tel_id in range(max_tel_id):
        if tel_id not in data.inst.pixel_pos:
            try:
                data.inst.pixel_pos[tel_id] \
                    = pyhessio.get_pixel_position(tel_id) * u.m                
            except HessioTelescopeIndexError:
                pass

    for tel_id in data.inst.pixel_pos:
        try:
            data.inst.optical_foclen[tel_id] \
                = pyhessio.get_optical_foclen(tel_id) * u.m
            data.inst.tel_pos[tel_id] \
                = pyhessio.get_telescope_position(tel_id) * u.m               
            nchans = pyhessio.get_num_channel(tel_id)
            npix = pyhessio.get_num_pixels(tel_id)
            nsamples = pyhessio.get_num_samples(tel_id)
            if nsamples <= 0: nsamples = 1
            data.inst.num_channels[tel_id] = nchans
            data.inst.num_pixels[tel_id] = npix
            data.inst.num_samples[tel_id] = nsamples
        except HessioGeneralError:
            pass
            

