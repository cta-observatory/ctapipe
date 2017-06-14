# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read HESSIO data.  

This requires the hessio python library to be installed
"""
import logging

from .containers import DataContainer
from ..core import Provenance

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
    logger.fatal(
        "the `pyhessio` python module is required to access MC data: {}"
        .format(err))
    raise err

__all__ = [
    'hessio_event_source',
]


def hessio_get_list_event_ids(url, max_events=None):
    """
    Faster method to get a list of all the event ids in the hessio file.
    This list can also be used to find out the number of events that exist
    in the file.

    Parameters
    ----------
    url : str
        path to file to open
    max_events : int, optional
        maximum number of events to read

    Returns
    -------
    event_id_list : list[num_events]
        A list with all the event ids that are in the file.

    """
    logger.warning("This method is slow. Need to find faster method.")
    try:
        with open_hessio(url) as pyhessio:
            Provenance().add_input_file(url)
            counter = 0
            event_id_list = []
            eventstream = pyhessio.move_to_next_event()
            for event_id in eventstream:
                event_id_list.append(event_id)
                counter += 1
                if max_events and counter >= max_events:
                    pyhessio.close_file()
                    break
            return event_id_list
    except HessioError:
        raise RuntimeError("hessio_event_source failed to open '{}'"
                           .format(url))


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

    with open_hessio(url) as pyhessio:
        # the container is initialized once, and data is replaced within
        # it after each yield
        Provenance().add_input_file(url)
        counter = 0
        eventstream = pyhessio.move_to_next_event()
        if allowed_tels is not None:
            allowed_tels = set(allowed_tels)
        data = DataContainer()
        data.meta['origin'] = "hessio"

        # some hessio_event_source specific parameters
        data.meta['input'] = url
        data.meta['max_events'] = max_events

        for event_id in eventstream:

            # Seek to requested event
            if requested_event is not None:
                current = counter
                if use_event_id:
                    current = event_id
                if not current == requested_event:
                    counter += 1
                    continue

            data.r0.run_id = pyhessio.get_run_number()
            data.r0.event_id = event_id
            data.r0.tels_with_data = set(pyhessio.get_teldata_list())
            data.r1.run_id = pyhessio.get_run_number()
            data.r1.event_id = event_id
            data.r1.tels_with_data = set(pyhessio.get_teldata_list())
            data.dl0.run_id = pyhessio.get_run_number()
            data.dl0.event_id = event_id
            data.dl0.tels_with_data = set(pyhessio.get_teldata_list())

            # handle telescope filtering by taking the intersection of
            # tels_with_data and allowed_tels
            if allowed_tels is not None:
                selected = data.r0.tels_with_data & allowed_tels
                if len(selected) == 0:
                    continue  # skip event
                data.r0.tels_with_data = selected
                data.r1.tels_with_data = selected
                data.dl0.tels_with_data = selected

            data.trig.tels_with_trigger \
                = pyhessio.get_central_event_teltrg_list()
            time_s, time_ns = pyhessio.get_central_event_gps_time()
            data.trig.gps_time = Time(time_s * u.s, time_ns * u.ns,
                                      format='unix', scale='utc')
            data.mc.energy = pyhessio.get_mc_shower_energy() * u.TeV
            data.mc.alt = Angle(pyhessio.get_mc_shower_altitude(), u.rad)
            data.mc.az = Angle(pyhessio.get_mc_shower_azimuth(), u.rad)
            data.mc.core_x = pyhessio.get_mc_event_xcore() * u.m
            data.mc.core_y = pyhessio.get_mc_event_ycore() * u.m
            first_int = pyhessio.get_mc_shower_h_first_int() * u.m
            data.mc.h_first_int = first_int

            # mc run header data
            data.mcheader.run_array_direction = \
                pyhessio.get_mc_run_array_direction()

            data.count = counter

            # this should be done in a nicer way to not re-allocate the
            # data each time (right now it's just deleted and garbage
            # collected)

            data.r0.tel.clear()
            data.r1.tel.clear()
            data.dl0.tel.clear()
            data.dl1.tel.clear()
            data.mc.tel.clear()  # clear the previous telescopes

            _fill_instrument_info(data, pyhessio)

            for tel_id in data.r0.tels_with_data:

                # event.mc.tel[tel_id] = MCCameraContainer()

                data.mc.tel[tel_id].dc_to_pe \
                    = pyhessio.get_calibration(tel_id)
                data.mc.tel[tel_id].pedestal \
                    = pyhessio.get_pedestal(tel_id)

                data.r0.tel[tel_id].adc_samples = \
                    pyhessio.get_adc_sample(tel_id)
                data.r0.tel[tel_id].adc_sums = \
                    pyhessio.get_adc_sum(tel_id)
                data.mc.tel[tel_id].reference_pulse_shape = \
                    pyhessio.get_ref_shapes(tel_id)

                nsamples = pyhessio.get_event_num_samples(tel_id)
                if nsamples <= 0:
                    nsamples = 1
                data.r0.tel[tel_id].num_samples = nsamples

                # load the data per telescope/pixel
                hessio_mc_npe = pyhessio.get_mc_number_photon_electron
                data.mc.tel[tel_id].photo_electron_image \
                    = hessio_mc_npe(telescope_id=tel_id)
                data.mc.tel[tel_id].meta['refstep'] = \
                    pyhessio.get_ref_step(tel_id)
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

            if max_events  and counter >= max_events:
                pyhessio.close_file()
                return


def _fill_instrument_info(data, pyhessio):
    """
    fill the data.inst structure with instrumental information.

    Parameters
    ----------
    data: DataContainer
        data container to fill in

    """
    if not data.inst.telescope_ids:
        data.inst.telescope_ids = list(pyhessio.get_telescope_ids())

        for tel_id in data.inst.telescope_ids:
            try:
                data.inst.pixel_pos[tel_id] \
                    = pyhessio.get_pixel_position(tel_id) * u.m
                data.inst.optical_foclen[tel_id] \
                    = pyhessio.get_optical_foclen(tel_id) * u.m
                data.inst.tel_pos[tel_id] \
                    = pyhessio.get_telescope_position(tel_id) * u.m
                nchans = pyhessio.get_num_channel(tel_id)
                npix = pyhessio.get_num_pixels(tel_id)
                data.inst.num_channels[tel_id] = nchans
                data.inst.num_pixels[tel_id] = npix
                data.inst.mirror_dish_area[tel_id] = \
                    pyhessio.get_mirror_area(tel_id) * u.m ** 2
                data.inst.mirror_numtiles[tel_id] = \
                    pyhessio.get_mirror_number(tel_id)
            except HessioGeneralError:
                pass
