from ctapipe.io import EventSource
import tables
import matplotlib.pyplot as plt
from ctapipe.io.datalevels import DataLevel

from ctapipe.instrument import (
    TelescopeDescription,
    SubarrayDescription,
    CameraDescription,
    CameraGeometry,
    CameraReadout,
    OpticsDescription,
)
from astropy.table import Table

from ctapipe.containers import EventAndMonDataContainer, EventType
from ctapipe.instrument.camera import UnknownPixelShapeWarning
from ctapipe.instrument.guess import guess_telescope, UNKNOWN_TELESCOPE
from ctapipe.containers import MCHeaderContainer
from ctapipe.containers import *
from ctapipe.containers import IntensityStatisticsContainer, PeakTimeStatisticsContainer
from ctapipe.io.eventsource import EventSource
from ctapipe.io.datalevels import DataLevel
from astropy.coordinates import Angle
import astropy.units as u


class DL1EventSource(EventSource):
    def __init__(
        self,
        input_url,
        config=None,
        parent=None,
        **kwargs
    ):
        """
        EventSource for dl1 files in the standard DL1 data format

        Parameters:
        -----------
        input_url : str
            Path of the file to load
        config: ??
        parent: ??
        kwargs
        """
        super().__init__(
            input_url=input_url,
            config=config,
            parent=parent,
            **kwargs
        )

        self.file_ = tables.open_file(input_url)
        self._subarray_info = self._prepare_subarray_info(
            self.file_.root.configuration.instrument
        )
        self._mc_header = self._parse_mc_header(
            self.file_.root.configuration.simulation.run
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.file_.close()

    @staticmethod
    def is_compatible(file_path):
        """
        Implementation needed!
        """
        return False

    @property
    def is_simulation(self):
        """
        Implementation needed!
        """
        return True

    @property
    def subarray(self):
        return self._subarray_info

    @property
    def datalevels(self):
        return (DataLevel.DL1_IMAGES, DataLevel.DL1_PARAMETERS)  ## need to check if they are in file?

    @property
    def obs_id(self):
        return set(self.file_.root.dl1.event.subarray.trigger.col("obs_id"))

    @property
    def mc_header(self):
        return self._mc_header

    def _generator(self):
        try:
            yield from self._generate_events()
        except EOFError:
            msg = 'EOFError reading from "{input_url}". Might be truncated'.format(
                input_url=self.input_url
            )
            self.log.warning(msg)
            warnings.warn(msg)

    def _prepare_subarray_info(self, instrument_description):
        """
        Constructs a SubArrayDescription object from
        self.file_.root.configuration.instrument.telescope.optics
        and 
        self.file_.root.configuration.instrument.subarray.layout
        tables.

        Returns
        -------
        SubarrayDescription :
            instrumental information
        """
        available_optics = instrument_description.telescope.optics.iterrows()
        available_telescopes = instrument_description.subarray.layout.iterrows()
    
        # The focal length choice is missing here
        # I am not sure how they are handled in the file
        # Will there be one of "e_fl" or "fl" in the columns?
        # This will only work if "e_fl" is available
        optic_descriptions = {}
        for optic in available_optics:
            optic_description = OpticsDescription(
                name=optic['name'].decode(),
                num_mirrors=optic['num_mirrors'],
                equivalent_focal_length=u.Quantity(
                    optic['equivalent_focal_length'],
                    u.m,
                ),
                mirror_area=u.Quantity(
                    optic['mirror_area'],
                    u.m ** 2,
                ),
                num_mirror_tiles=optic['num_mirror_tiles'],
            )
            optic_descriptions[optic['description'].decode()] = optic_description
        
        tel_positions = {}
        tel_descriptions = {}
        for telescope in available_telescopes:
            tel_positions[telescope['tel_id']] = (
                telescope['pos_x'],
                telescope['pos_y'],
                telescope['pos_z'],
            )
            geom = CameraGeometry.from_name(telescope['camera_type'].decode())
            optics = optic_descriptions[telescope['tel_description'].decode()]
            tel_descriptions[telescope['tel_id']] = TelescopeDescription(
                name=telescope['name'],
                tel_type=telescope['type'],
                optics=optics,
                camera=geom,
            )

        return SubarrayDescription(
            name='???',
            tel_positions=tel_positions,
            tel_descriptions=tel_descriptions,
        )

    def _parse_mc_header(self, header_table):
        """
        Construct a MCHeaderContainer from the
        self.file_.root.configuration.simulation.run
        """
        # We just assume there is only one row?
        # Thats kind of what the SimTelEventSource does
        # If mixed configurations are in one file, it gets problematic anyway, right? -> subarray layout?
        header = header_table.row
        return MCHeaderContainer(
                corsika_version=header["corsika_version"],
                simtel_version=header["simtel_version"],
                energy_range_min=header["energy_range_min"] * u.TeV,
                energy_range_max=header["energy_range_max"] * u.TeV,
                prod_site_B_total=header["prod_site_B_total"] * u.uT,
                prod_site_B_declination=Angle(header["prod_site_B_declination"], u.rad,),
                prod_site_B_inclination=Angle(header["prod_site_B_inclination"], u.rad,),
                prod_site_alt=header["prod_site_alt"] * u.m,
                spectral_index=header["spectral_index"],
                shower_prog_start=header["shower_prog_start"],
                shower_prog_id=header["shower_prog_id"],
                detector_prog_start=header["detector_prog_start"],
                detector_prog_id=header["detector_prog_id"],
                num_showers=header["num_showers"],
                shower_reuse=header["shower_reuse"],
                max_alt=header["max_alt"] * u.rad,
                min_alt=header["min_alt"] * u.rad,
                max_az=header["max_az"] * u.rad,
                min_az=header["min_az"] * u.rad,
                diffuse=header["diffuse"],
                max_viewcone_radius=header["max_viewcone_radius"] * u.deg,
                min_viewcone_radius=header["min_viewcone_radius"] * u.deg,
                max_scatter_range=header["max_scatter_range"] * u.m,
                min_scatter_range=header["min_scatter_range"] * u.m,
                core_pos_mode=header["core_pos_mode"],
                injection_height=header["injection_height"] * u.m,
                atmosphere=header["atmosphere"],
                corsika_iact_options=header["corsika_iact_options"],
                corsika_low_E_model=header["corsika_low_E_model"],
                corsika_high_E_model=header["corsika_high_E_model"],
                corsika_bunchsize=header["corsika_bunchsize"],
                corsika_wlen_min=header["corsika_wlen_min"] * u.nm,
                corsika_wlen_max=header["corsika_wlen_max"] * u.nm,
                corsika_low_E_detail=header["corsika_low_E_detail"],
                corsika_high_E_detail=header["corsika_high_E_detail"],
                run_array_direction=Angle(header["run_array_direction"] * u.rad),
            )

    def _generate_events(self):
        """
        Yield EventAndMonDataContainer to iterate through events.
        """
        data = EventAndMonDataContainer()
        data.meta["origin"] = "dl1 test" # Any Infos in the file?
        data.meta["input_url"] = self.input_url
        data.meta["max_events"] = self.max_events # Does this have any effect?
        data.mcheader = self._mc_header
        
        # loop array events
        for counter, array_event in enumerate(self.file_.root.dl1.event.subarray.trigger):
            # this should be done in a nicer way to not re-allocate the
            # data each time (right now it's just deleted and garbage
            # collected)
            data.r0.tel.clear()
            data.r1.tel.clear()
            data.dl0.tel.clear()
            data.dl1.tel.clear()
            data.mc.tel.clear()
            data.pointing.tel.clear()
            data.trigger.tel.clear()
            
            data.count = counter
            
            obs_id = array_event['obs_id']
            event_id = array_event['event_id']
            time = array_event['time']
            event_type = array_event['event_type']
            data.index.obs_id = obs_id
            data.index.event_id = event_id
            
            self._fill_trigger_info(data, array_event)
            self._fill_array_pointing(data, time)

            telescope_events = self.file_.root.dl1.event.telescope.trigger.where(
                f"(obs_id=={obs_id})&(event_id=={event_id})"
            )
            
            for telescope_event in telescope_events:
                obs_id = telescope_event['obs_id']
                event_id = telescope_event['event_id']
                teltrigger_time = telescope_event['telescopetrigger_time']
                tel_id = telescope_event['tel_id']
               
                self._fill_telescope_pointing(data, teltrigger_time)

                # bc of indexing in the table (1->001, 12->012, 123->123)
                index_id = ("000" + str(tel_id))[-3:]
                
                # where returns an iterator. Is there a way to directly get the row?
                mc_info_iterator = self.file_.root.simulation.event.subarray.shower.where(
                    f"(obs_id=={obs_id})&(event_id=={event_id})"
                )
                mc_info = validate_single_result_query(mc_info_iterator)
                if mc_info:
                    data.mc = MCEventContainer(
                        energy=mc_info['true_energy'],
                        alt=mc_info['true_alt'],
                        az=mc_info['true_az'],
                        core_x=mc_info['true_core_x'],
                        core_y=mc_info['true_core_y'],
                        h_first_int=mc_info['true_h_first_int'],
                        x_max=mc_info['true_x_max'],
                        shower_primary_id=mc_info['true_shower_primary_id'],
                    )
                
                dl1 = data.dl1.tel[tel_id]

                cam_iterator = self.file_.root.dl1.event.telescope.images[f'tel_{index_id}'].where(
                    f"(obs_id=={obs_id})&(event_id=={event_id})"
                )
                cam = validate_single_result_query(cam_iterator)
                if cam:
                    dl1.image = cam['image']
                    dl1.peak_time = cam['peak_time']
                    dl1.image_mask = cam['image_mask']
                
                    
                params_iterator = self.file_.root.dl1.event.telescope.parameters[f'tel_{index_id}'].where(
                    f"(obs_id=={obs_id})&(event_id=={event_id})"
                )
                params = validate_single_result_query(params_iterator)
                if params:
                    dl1.parameters.hillas.x = params['hillas_x']
                    dl1.parameters.hillas.y = params['hillas_y']
                    dl1.parameters.hillas.r = params['hillas_r']
                    dl1.parameters.hillas.phi = params['hillas_phi']
                    dl1.parameters.hillas.length = params['hillas_length']
                    dl1.parameters.hillas.width = params['hillas_width']
                    dl1.parameters.hillas.psi = params['hillas_psi']
                    dl1.parameters.hillas.skewness = params['hillas_skewness']
                    dl1.parameters.hillas.kurtosis = params['hillas_kurtosis']
                    dl1.parameters.hillas.intensity = params['hillas_intensity']

                    dl1.parameters.timing.slope = params['timing_slope']
                    dl1.parameters.timing.slope_err = params['timing_slope_err']
                    dl1.parameters.timing.intercept = params['timing_intercept']
                    dl1.parameters.timing.intercept_err = params['timing_intercept_err']
                    dl1.parameters.timing.deviation = params['timing_deviation']

                    dl1.parameters.leakage.pixels_width_1 = params['leakage_pixels_width_1']
                    dl1.parameters.leakage.pixels_width_2 = params['leakage_pixels_width_2']
                    dl1.parameters.leakage.intensity_width_1 = params['leakage_pixels_width_1']
                    dl1.parameters.leakage.intensity_width_2 = params['leakage_pixels_width_2']

                    dl1.parameters.concentration.cog = params['concentration_cog']
                    dl1.parameters.concentration.core = params['concentration_core']
                    dl1.parameters.concentration.pixel = params['concentration_pixel']

                    dl1.parameters.morphology.num_pixels = params['morphology_num_pixels']
                    dl1.parameters.morphology.num_islands = params['morphology_num_islands']
                    dl1.parameters.morphology.num_small_islands = params['morphology_num_small_islands']
                    dl1.parameters.morphology.num_medium_islands = params['morphology_num_medium_islands']
                    dl1.parameters.morphology.num_large_islands = params['morphology_num_large_islands']

                    dl1.parameters.intensity_statistics.max = params['intensity_max']
                    dl1.parameters.intensity_statistics.min = params['intensity_min']
                    dl1.parameters.intensity_statistics.mean = params['intensity_mean']
                    dl1.parameters.intensity_statistics.std = params['intensity_std']
                    dl1.parameters.intensity_statistics.skewness = params['intensity_skewness']
                    dl1.parameters.intensity_statistics.kurtosis = params['intensity_kurtosis']

                    dl1.parameters.peak_time_statistics.max = params['peak_time_max']
                    dl1.parameters.peak_time_statistics.min = params['peak_time_min']
                    dl1.parameters.peak_time_statistics.mean = params['peak_time_mean']
                    dl1.parameters.peak_time_statistics.std = params['peak_time_std']
                    dl1.parameters.peak_time_statistics.skewness = params['peak_time_skewness']
                    dl1.parameters.peak_time_statistics.kurtosis = params['peak_time_kurtosis']

            yield data

    def _fill_trigger_info(self, data, array_event):
        """
        Fill the trigger information for a provided array event.
        """
        obs_id = array_event['obs_id']
        event_id = array_event['event_id']
        array_trigger_iterator = self.file_.root.dl1.event.subarray.trigger.where(
            f"(obs_id=={obs_id})&(event_id=={event_id})"
        )
        array_trigger = validate_single_result_query(array_trigger_iterator)
        data.trigger.time = array_trigger['time']
        data.trigger.event_type = array_trigger['event_type']

        tel_trigger_iterator = self.file_.root.dl1.event.telescope.trigger.where(
            f"(obs_id=={obs_id})&(event_id=={event_id})"
        )
        tels_with_trigger = []
        for tel in tel_trigger_iterator:
            tels_with_trigger.append(tel['tel_id'])
            data.trigger.tel[tel['tel_id']].time = tel['telescopetrigger_time']
        data.trigger.tels_with_trigger = tels_with_trigger

    def _fill_array_pointing(self, data, time):
        """
        Implementaion needed
        """
        return None

    def _fill_telescope_pointing(self, data, teltrigger_time):
        """Implementation needed"""
        return None



def validate_single_result_query(row_iterator):
    """
    This is done in order to validate the row_iterator
    for where queries for which we expect to
    get one or zero rows. More than one row will raise an Exception
    
    Probably an unneeded hack and if its needed should maybe perform some
    logging.
    """
    query_result = [x for x in row_iterator]
    if len(query_result) == 1:
        return query_result[0]
    elif len(query_result) == 0:
        return None
    else:
        raise Exception("The query selected more than one row. This should not happen")
