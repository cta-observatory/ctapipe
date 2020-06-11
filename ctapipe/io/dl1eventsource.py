from ctapipe.io.datalevels import DataLevel
from ctapipe.instrument import (
    TelescopeDescription,
    SubarrayDescription,
    CameraGeometry,
    OpticsDescription,
)
from ctapipe.containers import EventAndMonDataContainer, EventType
from ctapipe.containers import MCHeaderContainer, MCEventContainer
from ctapipe.io.eventsource import EventSource
from ctapipe.io import HDF5TableReader
from astropy.coordinates import Angle
import astropy.units as u
import tables


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

        self.reader_ = HDF5TableReader(input_url)
        self._subarray_info = self._prepare_subarray_info(
            self.reader_._h5file.root.configuration.instrument
        )
        self._mc_header = self._parse_mc_header()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.file_.close()

    @staticmethod
    def is_compatible(file_path):
        with open(file_path) as f:
            magic_number = f.read(8)
        if magic_number != b'\x89HDF\r\n\x1a\n':
            return False
        else:
            hdf5file = tables.open_file(file_path)
            metadata = hdf5file.root._v_attrs
            organization = metadata.get('CTA CONTACT ORGANIZATION')
            if organization != 'CTA Consortium':
                return False
        return True

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
        return (DataLevel.DL1_IMAGES, DataLevel.DL1_PARAMETERS)

    @property
    def obs_id(self):
        return set(self.file_.root.dl1.event.subarray.trigger.col("obs_id"))

    @property
    def mc_header(self):
        return self._mc_header

    def _generator(self):
        yield from self._generate_events()

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

    def _parse_mc_header(self):
        """
        Construct a MCHeaderContainer from the
        self.file_.root.configuration.simulation.run
        """
        # We just assume there is only one row?
        # Thats kind of what the SimTelEventSource does
        # If mixed configurations are in one file, it gets problematic anyway, right? -> subarray layout?
        if 'simulation' in self.reader_._h5file.root.configuration.__members__:
            return next(self.reader_.read('/configuration/simulation/run', MCHeaderContainer()))

    def _generate_events(self):
        """
        Yield EventAndMonDataContainer to iterate through events.
        """
        data = EventAndMonDataContainer()
        #data.meta["origin"] = "dl1 test"  # Any Infos in the file?
        data.meta["input_url"] = self.input_url
        data.meta["max_events"] = self.max_events  # Does this have any effect?
        data.mcheader = self._mc_header

        # loop array events
        for counter, array_event in enumerate(self.reader_._h5file.root.dl1.event.subarray.trigger):
            # this should be done in a nicer way to not re-allocate the
            # data each time (right now it's just deleted and garbage
            # collected)
            data.dl1.tel.clear()
            data.mc.tel.clear()
            data.pointing.tel.clear()
            data.trigger.tel.clear()

            data.count = counter

            obs_id = array_event['obs_id']
            event_id = array_event['event_id']
            time = array_event['time']
            data.index.obs_id = obs_id
            data.index.event_id = event_id

            # where returns an iterator. Is there a way to directly get the row?
            if 'simulation' in self.reader_._h5file.root:
                data.mc = self.reader_.read(
                    '/simulation/event/subarray/shower',
                    MCHeaderContainer()
                )
            
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
                index_id = f'{tel_id:03d}'

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

        # Implementation needed
        event_type = array_event['event_type']
        data.trigger.event_type = EventType.UNKNOWN

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
    Maybe better, since its always the same query for obs id and event id:
    one function "select_event_rows" or similar
    """
    query_result = [x for x in row_iterator]
    if len(query_result) == 1:
        return query_result[0]
    elif len(query_result) == 0:
        return None
    else:
        raise Exception("The query selected more than one row. This should not happen")
