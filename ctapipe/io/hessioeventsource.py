from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time
from ctapipe.io.eventsource import EventSource
from ctapipe.io.simteleventsource import apply_simtel_r1_calibration
from ctapipe.io.containers import DataContainer
from ctapipe.instrument import (
    TelescopeDescription,
    SubarrayDescription,
    OpticsDescription,
    CameraGeometry,
)
from ctapipe.instrument.camera import UnknownPixelShapeWarning
from ctapipe.instrument.guess import guess_telescope, UNKNOWN_TELESCOPE
from ctapipe.calib.camera.gainselection import ThresholdGainSelector
import numpy as np
import warnings

__all__ = ['HESSIOEventSource']


class HESSIOEventSource(EventSource):
    _count = 0

    def __init__(self, config=None, parent=None, gain_selector=None, **kwargs):
        """
        EventSource for the hessio file format.

        This class utilises `pyhessio` to read the hessio file, and stores the
        information into the event containers.

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        gain_selector : ctapipe.calib.camera.gainselection.GainSelector
            The GainSelector to use. If None, then ThresholdGainSelector will be used.
        kwargs
        """
        super().__init__(config=config, parent=parent, **kwargs)

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

        # Waveforms from simtelarray have both gain channels
        # Gain selection is performed by this EventSource to produce R1 waveforms
        if gain_selector is None:
            gain_selector = ThresholdGainSelector(parent=self)
        self.gain_selector = gain_selector

    @staticmethod
    def is_compatible(file_path):
        '''This class should never be chosen in event_source()'''
        return False

    @property
    def subarray(self):
        with self.pyhessio.open_hessio(self.input_url) as file:
            next(file.move_to_next_event())
            return self._build_subarray_info(file)

    def __exit__(self, exc_type, exc_val, exc_tb):
        HESSIOEventSource._count -= 1
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

                    mc = data.mc.tel[tel_id]
                    r0 = data.r0.tel[tel_id]
                    r1 = data.r1.tel[tel_id]
                    pointing = data.pointing[tel_id]

                    adc_samples = file.get_adc_sample(tel_id)
                    if adc_samples.size == 0:
                        adc_samples = file.get_adc_sum(tel_id)[..., None]
                    dc_to_pe = file.get_calibration(tel_id)
                    pedestal = file.get_pedestal(tel_id)
                    r0.waveform = adc_samples
                    r1.waveform, r1.selected_gain_channel = apply_simtel_r1_calibration(
                        adc_samples, pedestal, dc_to_pe, self.gain_selector
                    )

                    mc.dc_to_pe = dc_to_pe
                    mc.pedestal = pedestal
                    r0.num_trig_pix = file.get_num_trig_pixels(tel_id)
                    r0.trig_pix_id = file.get_trig_pixels(tel_id)
                    mc.reference_pulse_shape = file.get_ref_shapes(tel_id)

                    # load the data per telescope/pixel
                    hessio_mc_npe = file.get_mc_number_photon_electron(tel_id)
                    mc.photo_electron_image = hessio_mc_npe
                    mc.meta['refstep'] = file.get_ref_step(tel_id)
                    mc.time_slice = file.get_time_slice(tel_id)
                    mc.azimuth_raw = file.get_azimuth_raw(tel_id)
                    mc.altitude_raw = file.get_altitude_raw(tel_id)
                    azimuth_cor = file.get_azimuth_cor(tel_id)
                    altitude_cor = file.get_altitude_cor(tel_id)

                    # hessioeventsource pass 0 if there is no altitude/azimuth correction
                    if azimuth_cor == 0 and mc.azimuth_raw != 0:
                        mc.azimuth_cor = np.nan
                        pointing.azimuth = u.Quantity(mc.azimuth_raw, u.rad)
                    else:
                        mc.azimuth_cor = azimuth_cor
                        pointing.azimuth = u.Quantity(azimuth_cor, u.rad)

                    if altitude_cor == 0 and mc.altitude_raw != 0:
                        mc.altitude_cor = np.nan
                        pointing.altitude = u.Quantity(mc.altitude_raw, u.rad)
                    else:
                        mc.altitude_cor = altitude_cor
                        pointing.altitude = u.Quantity(mc.altitude_cor, u.rad)

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
                tel = self._build_telescope_description(file, tel_id)
                tel_pos = u.Quantity(file.get_telescope_position(tel_id), u.m)
                subarray.tels[tel_id] = tel
                subarray.positions[tel_id] = tel_pos
            except self.pyhessio.HessioGeneralError:
                pass

        return subarray

    def _build_telescope_description(self, file, tel_id):
        pix_x, pix_y = u.Quantity(file.get_pixel_position(tel_id), u.m)
        focal_length = u.Quantity(file.get_optical_foclen(tel_id), u.m)
        n_pixels = len(pix_x)

        try:
            telescope = guess_telescope(n_pixels, focal_length)
        except ValueError:
            telescope = UNKNOWN_TELESCOPE

        pixel_shape = file.get_pixel_shape(tel_id)[0]
        try:
            pix_type, pix_rot = CameraGeometry.simtel_shape_to_type(pixel_shape)
        except ValueError:
            warnings.warn(
                f'Unkown pixel_shape {pixel_shape} for tel_id {tel_id}',
                UnknownPixelShapeWarning,
            )
            pix_type = 'hexagon'
            pix_rot = '0d'

        pix_area = u.Quantity(file.get_pixel_area(tel_id), u.m**2)

        mirror_area = u.Quantity(file.get_mirror_area(tel_id), u.m**2)
        num_tiles = file.get_mirror_number(tel_id)
        cam_rot = file.get_camera_rotation_angle(tel_id)
        num_mirrors = file.get_mirror_number(tel_id)
        sampling_rate = u.Quantity(1 / file.get_time_slice(tel_id), u.GHz)

        camera = CameraGeometry(
            telescope.camera_name,
            pix_id=np.arange(n_pixels),
            pix_x=pix_x,
            pix_y=pix_y,
            pix_area=pix_area,
            pix_type=pix_type,
            sampling_rate=sampling_rate,
            pix_rotation=pix_rot,
            cam_rotation=-Angle(cam_rot, u.rad),
            apply_derotation=True,
        )

        optics = OpticsDescription(
            name=telescope.name,
            num_mirrors=num_mirrors,
            equivalent_focal_length=focal_length,
            mirror_area=mirror_area,
            num_mirror_tiles=num_tiles,
        )

        return TelescopeDescription(name=telescope.name, tel_type=telescope.type,
                                    optics=optics, camera=camera)
