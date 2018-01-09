from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time
from ctapipe.core import Provenance
from ctapipe.io.eventfilereader import EventFileReader
from ctapipe.io.containers import DataContainer
from ctapipe.instrument import TelescopeDescription, SubarrayDescription


class HessioFileReader(EventFileReader):

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

        try:
            from pyhessio import open_hessio
            from pyhessio import HessioError
            from pyhessio import HessioTelescopeIndexError
            from pyhessio import HessioGeneralError
        except ImportError:
            msg = "The `pyhessio` python module is required to access MC data"
            self.log.error(msg)
            raise

        self.open_hessio = open_hessio
        self.HessioError = HessioError
        self.HessioTelescopeIndexError = HessioTelescopeIndexError
        self.HessioGeneralError = HessioGeneralError

        self.allowed_tels = None

    @staticmethod
    def is_compatible(file_path):
        return file_path.endswith('.gz')

    @property
    def camera(self):
        return 'hessio'

    def _generator(self):
        with self.open_hessio(self.input_path) as pyhessio:
            self.pyhessio = pyhessio
            # the container is initialized once, and data is replaced within
            # it after each yield
            Provenance().add_input_file(self.input_path, role='dl0.sub.evt')
            counter = 0
            eventstream = pyhessio.move_to_next_event()
            if self.allowed_tels is not None:
                self.allowed_tels = set(self.allowed_tels)
            data = DataContainer()
            data.meta['origin'] = "hessio"

            # some hessio_event_source specific parameters
            data.meta['input'] = self.input_path
            data.meta['max_events'] = self.max_events

            for event_id in eventstream:

                if counter == 0:
                    # subarray info is only available when an event is loaded,
                    # so load it on the first event.
                    data.inst.subarray = self._build_subarray_info(pyhessio)

                run_id = pyhessio.get_run_number()
                tels_with_data = set(pyhessio.get_teldata_list())
                data.count = counter
                data.r0.run_id = run_id
                data.r0.event_id = event_id
                data.r0.tels_with_data = tels_with_data
                data.r1.run_id = run_id
                data.r1.event_id = event_id
                data.r1.tels_with_data = tels_with_data
                data.dl0.run_id = run_id
                data.dl0.event_id = event_id
                data.dl0.tels_with_data = tels_with_data

                # handle telescope filtering by taking the intersection of
                # tels_with_data and allowed_tels
                if self.allowed_tels is not None:
                    selected = tels_with_data & self.allowed_tels
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
                data.mc.shower_primary_id = \
                    pyhessio.get_mc_shower_primary_id()

                # mc run header data
                data.mcheader.run_array_direction = \
                    pyhessio.get_mc_run_array_direction()

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

                    data.mc.tel[tel_id].dc_to_pe \
                        = pyhessio.get_calibration(tel_id)
                    data.mc.tel[tel_id].pedestal \
                        = pyhessio.get_pedestal(tel_id)

                    data.r0.tel[tel_id].adc_samples = \
                        pyhessio.get_adc_sample(tel_id)
                    if data.r0.tel[tel_id].adc_samples.size == 0:
                        # To handle ASTRI and dst files
                        data.r0.tel[tel_id].adc_samples = \
                            pyhessio.get_adc_sum(tel_id)[..., None]
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

                if self.max_events and counter >= self.max_events:
                    self.reset()
                    raise StopIteration
        self.reset()
        raise StopIteration

    def _build_subarray_info(self, pyhessio):
        """
        constructs a SubarrayDescription object from the info in an
        EventIO/HESSSIO file

        Parameters
        ----------
        pyhessio: HessioFile
            The open pyhessio file

        Returns
        -------
        SubarrayDescription :
            instrumental information
        """
        telescope_ids = list(pyhessio.get_telescope_ids())
        subarray = SubarrayDescription("MonteCarloArray")

        for tel_id in telescope_ids:
            try:

                pix_pos = pyhessio.get_pixel_position(tel_id) * u.m
                foclen = pyhessio.get_optical_foclen(tel_id) * u.m
                mirror_area = pyhessio.get_mirror_area(tel_id) * u.m ** 2
                num_tiles = pyhessio.get_mirror_number(tel_id)
                tel_pos = pyhessio.get_telescope_position(tel_id) * u.m

                tel = TelescopeDescription.guess(*pix_pos, foclen)
                tel.optics.mirror_area = mirror_area
                tel.optics.num_mirror_tiles = num_tiles
                subarray.tels[tel_id] = tel
                subarray.positions[tel_id] = tel_pos

            except self.HessioGeneralError:
                pass

        return subarray
