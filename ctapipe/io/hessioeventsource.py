from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time
from ctapipe.io.eventsource import EventSource
from ctapipe.io.containers import DataContainer
from ctapipe.instrument import TelescopeDescription, SubarrayDescription
import gzip
import struct

__all__ = ['HESSIOEventSource']


class HESSIOEventSource(EventSource):
    """
    EventSource for the hessio file format.

    This class utilises `pyhessio` to read the hessio file, and stores the
    information into the event containers.
    """
    _count = 0

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

        try:
            import pyhessio
        except ImportError:
            msg = "The `pyhessio` python module is required to access MC data"
            self.log.error(msg)
            raise

        self.pyhessio = pyhessio

        if HESSIOEventSource._count > 0:
            self.log.warning("Only one pyhessio event_source allowed at a time. "
                             "Previous hessio file will be closed.")
            self.pyhessio.close_file()
        HESSIOEventSource._count += 1

        self.metadata['is_simulation'] = True

    @staticmethod
    def is_compatible(file_path):
        # read the first 4 bytes
        with open(file_path, 'rb') as f:
            marker_bytes = f.read(4)
        # if file is gzip, read the first 4 bytes with gzip again
        if marker_bytes[0] == 0x1f and marker_bytes[1] == 0x8b:
            with gzip.open(file_path, 'rb') as f:
                marker_bytes = f.read(4)
        # check for the simtel magic marker
        int_marker, = struct.unpack('I', marker_bytes)
        return int_marker == 3558836791 or int_marker == 931798996

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pyhessio.close_file()

    def _generator(self):
        with self.pyhessio.open_hessio(self.input_url) as file:
            # the container is initialized once, and data is replaced within
            # it after each yield
            counter = 0
            eventstream = file.move_to_next_event()
            data = DataContainer()
            data.meta['origin'] = "hessio"

            # some hessio_event_source specific parameters
            data.meta['input_url'] = self.input_url
            data.meta['max_events'] = self.max_events

            for event_id in eventstream:

                if counter == 0:
                    # subarray info is only available when an event is loaded,
                    # so load it on the first event.
                    data.inst.subarray = self._build_subarray_info(file)

                obs_id = file.get_run_number()
                tels_with_data = set(file.get_teldata_list())
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

                data.trig.tels_with_trigger = (file.
                                               get_central_event_teltrg_list())
                time_s, time_ns = file.get_central_event_gps_time()
                data.trig.gps_time = Time(time_s * u.s, time_ns * u.ns,
                                          format='unix', scale='utc')
                data.mc.energy = file.get_mc_shower_energy() * u.TeV
                data.mc.alt = Angle(file.get_mc_shower_altitude(), u.rad)
                data.mc.az = Angle(file.get_mc_shower_azimuth(), u.rad)
                data.mc.core_x = file.get_mc_event_xcore() * u.m
                data.mc.core_y = file.get_mc_event_ycore() * u.m
                first_int = file.get_mc_shower_h_first_int() * u.m
                data.mc.h_first_int = first_int
                data.mc.x_max = file.get_mc_shower_xmax() * u.g / (u.cm**2)
                data.mc.shower_primary_id = file.get_mc_shower_primary_id()

                # mc run header data
                data.mcheader.run_array_direction = Angle(
                    file.get_mc_run_array_direction() * u.rad
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

                    # event.mc.tel[tel_id] = MCCameraContainer()

                    data.mc.tel[tel_id].dc_to_pe = file.get_calibration(tel_id)
                    data.mc.tel[tel_id].pedestal = file.get_pedestal(tel_id)
                    data.r0.tel[tel_id].waveform = (file.
                                                    get_adc_sample(tel_id))
                    if data.r0.tel[tel_id].waveform.size == 0:
                        # To handle ASTRI and dst files
                        data.r0.tel[tel_id].waveform = (file.
                                                        get_adc_sum(tel_id)[..., None])
                    data.r0.tel[tel_id].image = file.get_adc_sum(tel_id)
                    data.r0.tel[tel_id].num_trig_pix = file.get_num_trig_pixels(tel_id)
                    data.r0.tel[tel_id].trig_pix_id = file.get_trig_pixels(tel_id)
                    data.mc.tel[tel_id].reference_pulse_shape = (file.
                                                                 get_ref_shapes(tel_id))

                    nsamples = file.get_event_num_samples(tel_id)
                    if nsamples <= 0:
                        nsamples = 1
                    data.r0.tel[tel_id].num_samples = nsamples

                    # load the data per telescope/pixel
                    hessio_mc_npe = file.get_mc_number_photon_electron(tel_id)
                    data.mc.tel[tel_id].photo_electron_image = hessio_mc_npe
                    data.mc.tel[tel_id].meta['refstep'] = (file.
                                                           get_ref_step(tel_id))
                    data.mc.tel[tel_id].time_slice = (file.
                                                      get_time_slice(tel_id))
                    data.mc.tel[tel_id].azimuth_raw = (file.
                                                       get_azimuth_raw(tel_id))
                    data.mc.tel[tel_id].altitude_raw = (file.
                                                        get_altitude_raw(tel_id))
                    data.mc.tel[tel_id].azimuth_cor = (file.
                                                       get_azimuth_cor(tel_id))
                    data.mc.tel[tel_id].altitude_cor = (file.
                                                        get_altitude_cor(tel_id))
                yield data
                counter += 1

        return

    def _build_subarray_info(self, file):
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
        telescope_ids = list(file.get_telescope_ids())
        subarray = SubarrayDescription("MonteCarloArray")

        for tel_id in telescope_ids:
            try:

                pix_pos = file.get_pixel_position(tel_id) * u.m
                foclen = file.get_optical_foclen(tel_id) * u.m
                mirror_area = file.get_mirror_area(tel_id) * u.m ** 2
                num_tiles = file.get_mirror_number(tel_id)
                tel_pos = file.get_telescope_position(tel_id) * u.m

                tel = TelescopeDescription.guess(*pix_pos,
                                                 equivalent_focal_length=foclen)
                tel.optics.mirror_area = mirror_area
                tel.optics.num_mirror_tiles = num_tiles
                subarray.tels[tel_id] = tel
                subarray.positions[tel_id] = tel_pos

            except self.pyhessio.HessioGeneralError:
                pass

        return subarray
