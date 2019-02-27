import warnings
import numpy as np
from ctapipe.io.eventsource import EventSource
from ctapipe.io.containers import DataContainer
from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time
from ctapipe.instrument import TelescopeDescription, SubarrayDescription
from traitlets import Bool


from eventio.simtel.simtelfile import SimTelFile
from eventio.file_types import is_eventio

__all__ = ['SimTelEventSource']


class SimTelEventSource(EventSource):
    skip_calibration_events = Bool(True, help='Skip calibration events').tag(config=True)

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

        self.metadata['is_simulation'] = True

        # traitlets creates an empty set as default,
        # which ctapipe treats as no restriction on the telescopes
        # but eventio treats an emty set as "no telescopes allowed"
        # so we explicitly pass None in that case
        self.file_ = SimTelFile(
            self.input_url,
            allowed_telescopes=self.allowed_tels if self.allowed_tels else None,
            skip_calibration=self.skip_calibration_events
        )

        self._subarray_info = self.prepare_subarray_info(
            self.file_.telescope_descriptions,
            self.file_.header
        )

    @staticmethod
    def prepare_subarray_info(telescope_descriptions, header):
        """
        Constructs a SubarrayDescription object from the
        ``telescope_descriptions`` given by ``SimTelFile``

        Parameters
        ----------
        telescope_descriptions: dict
            telescope descriptions as given by ``SimTelFile.telescope_descriptions``
        header: dict
            header as returned by ``SimTelFile.header``

        Returns
        -------
        SubarrayDescription :
            instrumental information
        """

        tel_descriptions = {}  # tel_id : TelescopeDescription
        tel_positions = {}  # tel_id : TelescopeDescription

        for tel_id, telescope_description in telescope_descriptions.items():
            cam_settings = telescope_description['camera_settings']
            tel_description = TelescopeDescription.guess(
                cam_settings['pixel_x'] * u.m,
                cam_settings['pixel_y'] * u.m,
                equivalent_focal_length=cam_settings['focal_length'] * u.m
            )
            tel_description.optics.mirror_area = (
                cam_settings['mirror_area'] * u.m ** 2
            )
            tel_description.optics.num_mirror_tiles = (
                cam_settings['n_mirrors']
            )
            tel_descriptions[tel_id] = tel_description

            tel_idx = np.where(header['tel_id'] == tel_id)[0][0]
            tel_positions[tel_id] = header['tel_pos'][tel_idx] * u.m

        return SubarrayDescription(
            "MonteCarloArray",
            tel_positions=tel_positions,
            tel_descriptions=tel_descriptions,
        )

    @staticmethod
    def is_compatible(file_path):
        return is_eventio(file_path)

    def _generator(self):
        try:
            yield from self.__generator()
        except EOFError:
            msg = 'EOFError reading from "{input_url}". Might be truncated'.format(
                input_url=self.input_url
            )
            self.log.warning(msg)
            warnings.warn(msg)

    def __generator(self):
        data = DataContainer()
        data.meta['origin'] = 'hessio'
        data.meta['input_url'] = self.input_url
        data.meta['max_events'] = self.max_events

        for counter, array_event in enumerate(self.file_):
            # next lines are just for debugging
            self.array_event = array_event
            data.event_type = array_event['type']

            # calibration events do not have an event id
            if data.event_type == 'calibration':
                event_id = -1
            else:
                event_id = array_event['event_id']

            data.inst.subarray = self._subarray_info

            obs_id = self.file_.header['run']
            tels_with_data = set(array_event['telescope_events'].keys())
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

            trigger_information = array_event['trigger_information']

            data.trig.tels_with_trigger = trigger_information['triggered_telescopes']
            time_s, time_ns = trigger_information['gps_time']
            data.trig.gps_time = Time(time_s * u.s, time_ns * u.ns,
                                      format='unix', scale='utc')

            if data.event_type == 'data':
                self.fill_mc_information(data, array_event)

            # this should be done in a nicer way to not re-allocate the
            # data each time (right now it's just deleted and garbage
            # collected)
            data.r0.tel.clear()
            data.r1.tel.clear()
            data.dl0.tel.clear()
            data.dl1.tel.clear()
            data.mc.tel.clear()  # clear the previous telescopes

            telescope_events = array_event['telescope_events']
            tracking_positions = array_event['tracking_positions']
            for tel_id, telescope_event in telescope_events.items():
                tel_index = self.file_.header['tel_id'].tolist().index(tel_id)
                telescope_description = self.file_.telescope_descriptions[tel_id]

                data.mc.tel[tel_id].dc_to_pe = array_event['laser_calibrations'][tel_id]['calib']
                data.mc.tel[tel_id].pedestal = array_event['camera_monitorings'][tel_id]['pedestal']
                adc_samples = telescope_event.get('adc_samples')
                if adc_samples is None:
                    adc_samples = telescope_event['adc_sums'][:, :, np.newaxis]
                data.r0.tel[tel_id].waveform = adc_samples
                data.r0.tel[tel_id].num_samples = adc_samples.shape[-1]
                # We should not calculate stuff in an event source
                # if this is not needed, we calculate it for nothing
                data.r0.tel[tel_id].image = adc_samples.sum(axis=-1)

                pixel_lists = telescope_event['pixel_lists']
                data.r0.tel[tel_id].num_trig_pix = pixel_lists.get(0, {'pixels': 0})['pixels']
                if data.r0.tel[tel_id].num_trig_pix > 0:
                    data.r0.tel[tel_id].trig_pix_id = pixel_lists[0]['pixel_list']

                pixel_settings = telescope_description['pixel_settings']
                data.mc.tel[tel_id].reference_pulse_shape = pixel_settings['ref_shape'].astype('float64')
                data.mc.tel[tel_id].meta['refstep'] = float(pixel_settings['ref_step'])
                data.mc.tel[tel_id].time_slice = float(pixel_settings['time_slice'])

                n_pixel = data.r0.tel[tel_id].waveform.shape[-2]
                data.mc.tel[tel_id].photo_electron_image = (
                    array_event.get('photoelectrons', {})
                               .get(tel_index, {})
                               .get('photoelectrons', np.zeros(n_pixel, dtype='float32'))
                )

                tracking_position = tracking_positions[tel_id]
                data.mc.tel[tel_id].azimuth_raw = tracking_position['azimuth_raw']
                data.mc.tel[tel_id].altitude_raw = tracking_position['altitude_raw']
                data.mc.tel[tel_id].azimuth_cor = tracking_position.get('azimuth_cor', 0)
                data.mc.tel[tel_id].altitude_cor = tracking_position.get('altitude_cor', 0)
            yield data

    def fill_mc_information(self, data, array_event):
        mc_event = array_event['mc_event']
        mc_shower = array_event['mc_shower']

        data.mc.energy = mc_shower['energy'] * u.TeV
        data.mc.alt = Angle(mc_shower['altitude'], u.rad)
        data.mc.az = Angle(mc_shower['azimuth'], u.rad)
        data.mc.core_x = mc_event['xcore'] * u.m
        data.mc.core_y = mc_event['ycore'] * u.m
        first_int = mc_shower['h_first_int'] * u.m
        data.mc.h_first_int = first_int
        data.mc.x_max = mc_shower['xmax'] * u.g / (u.cm**2)
        data.mc.shower_primary_id = mc_shower['primary_id']

        # mc run header data
        data.mcheader.run_array_direction = Angle(
            self.file_.header['direction'] * u.rad
        )
        mc_run_head = self.file_.mc_run_headers[-1]
        data.mcheader.corsika_version = mc_run_head['shower_prog_vers']
        data.mcheader.simtel_version = mc_run_head['detector_prog_vers']
        data.mcheader.energy_range_min = mc_run_head['E_range'][0] * u.TeV
        data.mcheader.energy_range_max = mc_run_head['E_range'][1] * u.TeV
        data.mcheader.prod_site_B_total = mc_run_head['B_total'] * u.uT
        data.mcheader.prod_site_B_declination = Angle(
            mc_run_head['B_declination'] * u.rad)
        data.mcheader.prod_site_B_inclination = Angle(
            mc_run_head['B_inclination'] * u.rad)
        data.mcheader.prod_site_alt = mc_run_head['obsheight'] * u.m
        data.mcheader.spectral_index = mc_run_head['spectral_index']
        data.mcheader.shower_prog_start = mc_run_head['shower_prog_start']
        data.mcheader.shower_prog_id = mc_run_head['shower_prog_id']
        data.mcheader.detector_prog_start = mc_run_head['detector_prog_start']
        data.mcheader.detector_prog_id = mc_run_head['detector_prog_id']
        data.mcheader.num_showers = mc_run_head['n_showers']
        data.mcheader.shower_reuse = mc_run_head['n_use']
        data.mcheader.max_alt = mc_run_head['alt_range'][1] * u.rad
        data.mcheader.min_alt = mc_run_head['alt_range'][0] * u.rad
        data.mcheader.max_az = mc_run_head['az_range'][1] * u.rad
        data.mcheader.min_az = mc_run_head['az_range'][0] * u.rad
        data.mcheader.diffuse = mc_run_head['diffuse']
        data.mcheader.max_viewcone_radius = mc_run_head['viewcone'][1] * u.deg
        data.mcheader.min_viewcone_radius = mc_run_head['viewcone'][0] * u.deg
        data.mcheader.max_scatter_range = mc_run_head['core_range'][1] * u.m
        data.mcheader.min_scatter_range = mc_run_head['core_range'][0] * u.m
        data.mcheader.core_pos_mode = mc_run_head['core_pos_mode']
        data.mcheader.injection_height = mc_run_head['injection_height'] * u.m
        data.mcheader.atmosphere = mc_run_head['atmosphere']
        data.mcheader.corsika_iact_options = mc_run_head['corsika_iact_options']
        data.mcheader.corsika_low_E_model = mc_run_head['corsika_low_E_model']
        data.mcheader.corsika_high_E_model = mc_run_head['corsika_high_E_model']
        data.mcheader.corsika_bunchsize = mc_run_head['corsika_bunchsize']
        data.mcheader.corsika_wlen_min = mc_run_head['corsika_wlen_min'] * u.nm
        data.mcheader.corsika_wlen_max = mc_run_head['corsika_wlen_max'] * u.nm
        data.mcheader.corsika_low_E_detail = mc_run_head['corsika_low_E_detail']
        data.mcheader.corsika_high_E_detail = mc_run_head['corsika_high_E_detail']

