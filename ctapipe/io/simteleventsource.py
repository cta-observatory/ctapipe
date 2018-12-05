import numpy as np
from ctapipe.io.eventsource import EventSource
from ctapipe.io.hessioeventsource import HESSIOEventSource
from ctapipe.io.containers import DataContainer
from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time
from ctapipe.instrument import TelescopeDescription, SubarrayDescription

from eventio.simtel.simtelfile import SimTelFile

__all__ = ['SimTelEventSource']

import eventio

class SimTelEventSource(EventSource):

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        self.metadata['is_simulation'] = True
        self.file_ = SimTelFile(self.input_url)

        self._subarray_info = self.prepare_subarray_info()

    def prepare_subarray_info(self):
        """
        constructs a SubarrayDescription object from the info in an
        EventIO/HESSSIO file

        Parameters
        ----------
        file: HessioFile
            The open pyhessio file

        Returns
        -------
        SubarrayDescription :
            instrumental information
        """

        subarray = SubarrayDescription("MonteCarloArray")

        for tel_id, cam_settings in self.file_.cam_settings.items():
            tel = TelescopeDescription.guess(
                cam_settings['pixel_x'] * u.m,
                cam_settings['pixel_y'] * u.m,
                equivalent_focal_length=cam_settings['focal_length'] * u.m
            )
            tel.optics.mirror_area = cam_settings['mirror_area'] * u.m ** 2
            tel.optics.num_mirror_tiles = cam_settings['mirror_area']
            subarray.tels[tel_id] = tel
            H = self.file_.header
            subarray.positions[tel_id] = H['tel_pos'][H['tel_id'] - 1] * u.m

        return subarray

    @staticmethod
    def is_compatible(file_path):
        # can be copied here verbatim if HESSIOEventSource should be removed
        return HESSIOEventSource.is_compatible(file_path)

    def _generator(self):


        data = DataContainer()
        data.meta['origin'] = "eventio"
        data.meta['input_url'] = self.input_url
        data.meta['max_events'] = self.max_events

        for counter, (shower, event) in enumerate(self.file_):
            # next lines are just for debugging
            self._event = event
            self._shower = shower

            event_id = event['pe_sum']['event']
            data.inst.subarray = self._subarray_info

            obs_id = self.file_.header['run']
            tels_with_data = set(event['event']['tel_events'].keys())
            data.count = counter
            data.r0.obs_id = obs_id
            data.r0.event_id = event_id
            data.r0.tels_with_data = tels_with_data
            data.r1.obs_id = obs_id
            data.r1.event_id = event_id
            data.r1.tels_with_data = tels_with_data
            data.dl0.obs_id = obs_id
            data.dl0.event_id = event_id
            data.dl0.tels_with_data = tels_with_data

            # handle telescope filtering by taking the intersection of
            # tels_with_data and allowed_tels
            if len(self.allowed_tels) > 0:
                selected = tels_with_data & self.allowed_tels
                if len(selected) == 0:
                    continue  # skip event
                data.r0.tels_with_data = selected
                data.r1.tels_with_data = selected
                data.dl0.tels_with_data = selected

            cent_event = event['event']['cent_event']
            MC_EVENT = event['mc_event']

            data.trig.tels_with_trigger = cent_event['triggered_telescopes']
            time_s, time_ns = cent_event['gps_time']
            data.trig.gps_time = Time(time_s * u.s, time_ns * u.ns,
                                      format='unix', scale='utc')

            data.mc.energy = shower['energy'] * u.TeV
            data.mc.alt = Angle(shower['altitude'], u.rad)
            data.mc.az = Angle(shower['azimuth'], u.rad)
            data.mc.core_x = MC_EVENT['xcore'] * u.m
            data.mc.core_y = MC_EVENT['ycore'] * u.m
            first_int = shower['h_first_int'] * u.m
            data.mc.h_first_int = first_int
            data.mc.x_max = shower['xmax'] * u.g / (u.cm**2)
            data.mc.shower_primary_id = shower['primary_id']

            # mc run header data
            data.mcheader.run_array_direction = Angle(
                self.file_.header['direction'] * u.rad
            )

            # this should be done in a nicer way to not re-allocate the
            # data each time (right now it's just deleted and garbage
            # collected)
            data.r0.tel.clear()
            data.r1.tel.clear()
            data.dl0.tel.clear()
            data.dl1.tel.clear()
            data.mc.tel.clear()  # clear the previous telescopes

            for tel_id in tels_with_data:
                H = event['event']['tel_events'][tel_id]['header']
                PL = event['event']['tel_events'][tel_id]['pixel_list']

                data.mc.tel[tel_id].dc_to_pe = self.file_.lascal[tel_id]['calib']
                data.mc.tel[tel_id].pedestal = self.file_.tel_moni[tel_id]['pedestal']
                data.r0.tel[tel_id].waveform = event['event']['tel_events'][tel_id]['waveform']
                data.r0.tel[tel_id].num_samples = data.r0.tel[tel_id].waveform.shape[-1]
                # We should not calculate stuff in an event source
                # if this is not needed, we calculate it for nothing
                data.r0.tel[tel_id].image = data.r0.tel[tel_id].waveform.sum(axis=-1)
                data.r0.tel[tel_id].num_trig_pix = len(PL['pixel_list'])
                data.r0.tel[tel_id].trig_pix_id = PL['pixel_list']
                data.mc.tel[tel_id].reference_pulse_shape = self.file_.ref_pulse[tel_id]['shape']

                n_pixel = data.r0.tel[tel_id].waveform.shape[-2]

                data.mc.tel[tel_id].photo_electron_image = np.zeros((n_pixel, ), dtype='i2')
                # photo_electron_image needs to be read from 1205 objects
                data.mc.tel[tel_id].meta['refstep'] = self.file_.ref_pulse[tel_id]['step']
                data.mc.tel[tel_id].time_slice = self.file_.time_slices_per_telescope[tel_id]

                data.mc.tel[tel_id].azimuth_raw = event['event']['tel_events'][tel_id]['track']['azimuth_raw']
                data.mc.tel[tel_id].altitude_raw = event['event']['tel_events'][tel_id]['track']['altitude_raw']
                try:
                    data.mc.tel[tel_id].azimuth_cor = event['event']['tel_events'][tel_id]['track']['azimuth_cor']
                    data.mc.tel[tel_id].altitude_cor = event['event']['tel_events'][tel_id]['track']['altitude_cor']
                except KeyError:
                    data.mc.tel[tel_id].azimuth_cor = 0
                    data.mc.tel[tel_id].altitude_cor = 0
            yield data
