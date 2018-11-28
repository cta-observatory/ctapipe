

import glob
import re

import h5py
import numpy as np
import scipy.interpolate

from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time
from ctapipe.io.eventsource import EventSource
from ctapipe.io.containers import DataContainer, TelescopePointingContainer, WeatherContainer
from ctapipe.instrument import TelescopeDescription, SubarrayDescription, OpticsDescription, CameraGeometry
import gzip
import struct


__all__ = ['MAGICEventSourceHDF5', 'MAGICEventSourceROOT']


class MAGICEventSourceHDF5(EventSource):
    """
    EventSource for MAGIC raw data converted to hdf5.

    This class utilises `h5py` to read the hdf5 file, and stores the
    information into the event containers.
    """
    _count = 0

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

        self.h5py = h5py
        self.file = h5py.File(self.input_url)

    @staticmethod
    def is_compatible(file_path):
        import h5py
        # check general format:
        if not h5py.is_hdf5(file_path):
            return False
        # crude check if hdf5 file contains MAGIC raw data:
        with h5py.File(file_path, "r") as file:
            intrument_attr = file.attrs['instrument']
            if intrument_attr.tostring() == b"MAGIC":
                return True
            else:
                return False
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def _generator(self):
        with self.h5py.File(self.input_url, "r") as input_file:
            # the container is initialized once, and data is replaced within
            # it after each yield
            counter = 0
            #eventstream = input_file.move_to_next_event()
            data = DataContainer()
            data.meta['origin'] = "MAGIC"

            # some hessio_event_source specific parameters
            data.meta['input_url'] = self.input_url
            data.meta['max_events'] = self.max_events

            events_per_tel = [[], []]

            if not input_file.attrs['RunType'] == b"Data":
                data.meta['is_simulation'] = True
                dt = np.float
            else:
                data.meta['is_simulation'] = False
                dt = np.int16
                
            # check which telescopes have data:
            
            if input_file.attrs['dl_export'] == b"dl1":
                eventstream = input_file['dl1/event_id']
                if input_file.attrs['data format'] == b"mono M1":
                    tels_in_file = ["M1"]
                elif input_file.attrs['data format'] == b"mono M2":
                    tels_in_file = ["M2"]
                elif input_file.attrs['data format'] == b"stereo":
                    tels_in_file = ["M1", "M2"]
                
            elif input_file.attrs['dl_export'] == b"r0":
                if list(input_file.keys()) == ["MAGIC1_rawdata"]:
                    eventstream = input_file['MAGIC1_rawdata/EvtHeader/StereoEvtNumber']
                    events_per_tel[0] = np.array(eventstream, dtype = dt)
                    tels_in_file = [1]
                elif list(input_file.keys()) == ["MAGIC2_rawdata"]:
                    eventstream = input_file['MAGIC2_rawdata/EvtHeader/StereoEvtNumber']
                    events_per_tel[1] = np.array(eventstream, dtype = dt)
                    tels_in_file = [2]
                elif list(input_file.keys()) == ["MAGIC1_rawdata", "MAGIC2_rawdata"]:
                    events_per_tel[0] = np.array(input_file['MAGIC1_rawdata/EvtHeader/StereoEvtNumber'], dtype = dt)
                    events_per_tel[1] = np.array(input_file['MAGIC2_rawdata/EvtHeader/StereoEvtNumber'], dtype = dt)
                    tels_in_file = [1, 2]
                    # order MC mono events into event stream:
                    if data.meta['is_simulation'] == True:
                        for i_tel in range(2):
                            for i_event in range(len(events_per_tel[i_tel])):
                                if events_per_tel[i_tel][i_event] == 0:
                                    if i_event != 0:
                                        events_per_tel[i_tel][i_event] = np.random.uniform(events_per_tel[i_tel][i_event - 1], np.floor(events_per_tel[i_tel][i_event - 1]) + 1)
                                    else:
                                        events_per_tel[i_tel][i_event] = np.random.uniform(0, 1)
                    eventstream = np.union1d(events_per_tel[0], events_per_tel[1])
                    
            else:
                raise IOError("MAGIC data level attribute 'dl_export' not found in input_file (value should be 'r0' or 'dl1').")
            
            optics = OpticsDescription.from_name(str(input_file['inst/subarray'].attrs['OpticsDescription'])[2:-1])
            geom = CameraGeometry.from_name(str(input_file['inst/subarray'].attrs['CameraGeometry'])[2:-1])
            magic_tel_description = TelescopeDescription(optics=optics, camera=geom)
            magic_tel_descriptions = {1: magic_tel_description, 2: magic_tel_description}
            # magic_tel_positions = {1: [input_file['inst/subarray/tel_coords']['M1'][0]*u.m, input_file['inst/subarray/tel_coords']['M1'][1]*u.m, input_file['inst/subarray/tel_coords']['M1'][2]*u.m],
            #                        2: [input_file['inst/subarray/tel_coords']['M2'][0]*u.m, input_file['inst/subarray/tel_coords']['M2'][1]*u.m, input_file['inst/subarray/tel_coords']['M2'][2]*u.m]}

            magic_tel_positions = {1: [input_file['inst/subarray/tel_coords']['M1'][0], input_file['inst/subarray/tel_coords']['M1'][1], input_file['inst/subarray/tel_coords']['M1'][2]]*u.m,
                                   2: [input_file['inst/subarray/tel_coords']['M2'][0], input_file['inst/subarray/tel_coords']['M2'][1], input_file['inst/subarray/tel_coords']['M2'][2]]*u.m}



            magic_subarray = SubarrayDescription(str(input_file.attrs['instrument'])[2:-1], magic_tel_positions, magic_tel_descriptions)

            
            for i_event, event_id in enumerate(eventstream):

#                 if counter == 0:
#                     # subarray info is only available when an event is loaded,
#                     # so load it on the first event.
#                     data.inst.subarray = self._build_subarray_info(input_file)
# 
                obs_id = input_file.attrs['RunNumber']
                data.count = counter
                data.r0.obs_id = input_file.attrs['RunNumber']
                data.r0.event_id = event_id
                data.r1.obs_id = input_file.attrs['RunNumber']
                data.r1.event_id = event_id
                data.dl0.obs_id = obs_id
                data.dl0.event_id = event_id
# 
#                 # handle telescope filtering by taking the intersection of
#                 # tels_with_data and allowed_tels
#                 if len(self.allowed_tels) > 0:
#                     selected = tels_with_data & self.allowed_tels
#                     if len(selected) == 0:
#                         continue  # skip event
#                     data.r0.tels_with_data = selected
#                     data.r1.tels_with_data = selected
#                     data.dl0.tels_with_data = selected
# 
#                 data.trig.tels_with_trigger = (input_file.
#                                                get_central_event_teltrg_list())
#                 time_s, time_ns = input_file.get_central_event_gps_time()
#                 data.trig.gps_time = Time(time_s * u.s, time_ns * u.ns,
#                                           format='unix', scale='utc')

#                 data.mc.alt = Angle(input_file.get_mc_shower_altitude(), u.rad)
#                 data.mc.az = Angle(input_file.get_mc_shower_azimuth(), u.rad)

#                 data.mc.x_max = input_file.get_mc_shower_xmax() * u.g / (u.cm**2)
#                 data.mc.shower_primary_id = input_file.get_mc_shower_primary_id()
# 
#                 # mc run header data
#                 data.mcheader.run_array_direction = Angle(
#                     input_file.get_mc_run_array_direction() * u.rad
#                 )
# 
#                 # this should be done in a nicer way to not re-allocate the
#                 # data each time (right now it's just deleted and garbage
#                 # collected)
# 
                data.r0.tel.clear()
                data.r1.tel.clear()
                data.dl0.tel.clear()
                data.dl1.tel.clear()
                data.mc.tel.clear()  # clear the previous telescopes

                tels_with_data_tmp = np.zeros(2)

                for i_tel, tel_id in enumerate(tels_in_file):

                    # load r0 data:
                    if input_file.attrs['dl_export'] == b"r0":
                        # search event
                        if event_id not in events_per_tel[i_tel]:
                            nevent = -1
                        else:
                            nevent = np.searchsorted(events_per_tel[i_tel], event_id, side='left')
                            tels_with_data_tmp[i_tel] = 1
                            
                        # sort out remaining calibration runs:
                        if input_file['MAGIC'+str(tel_id) +'_rawdata/EvtHeader/TrigPattern']["L3 trigger"][nevent] == False:
                            nevent = -1
                            tels_with_data_tmp[i_tel] = 0
    #                     # event.mc.tel[tel_id] = MCCameraContainer()
    # 
    #                     data.mc.tel[tel_id].dc_to_pe = input_file.get_calibration(tel_id)
    #                     data.mc.tel[tel_id].pedestal = input_file.get_pedestal(tel_id)
                        if nevent == -1:
                            data.r0.tel[tel_id].waveform = None
                        else:
                            data.r0.tel[tel_id].waveform = input_file['MAGIC'+str(tel_id) +'_rawdata/Events'][...,nevent]
    
                        data.r0.tel[tel_id].image = np.sum(data.r0.tel[tel_id].waveform, axis=0)
                        data.r0.tel[tel_id].num_trig_pix = input_file['MAGIC'+str(tel_id) +'_rawdata/EvtHeader/NumTrigLvl2'][nevent]
                        
                        # add MC information:
                        if data.meta['is_simulation'] == True:
                            # energy of event should be the same in both telescopes, so simply try both:
                            data.mc.energy = input_file['MAGIC'+str(tel_id) +'_rawdata/McHeader/Energy']["Energy"][nevent] * u.TeV
                            data.mc.core_x = input_file['MAGIC'+str(tel_id) +'_rawdata/McHeader/Core_xy']["Core_x"][nevent] * u.m
                            data.mc.core_y = input_file['MAGIC'+str(tel_id) +'_rawdata/McHeader/Core_xy']["Core_y"][nevent] * u.m
                            data.mc.h_first_int = input_file['MAGIC'+str(tel_id) +'_rawdata/McHeader/H_first_int']["H_first_int"][nevent] * u.m
                        
    #                     data.r0.tel[tel_id].trig_pix_id = input_file.get_trig_pixels(tel_id)
    #                     data.mc.tel[tel_id].reference_pulse_shape = (input_file.
    #                                                                  get_ref_shapes(tel_id))
    #  
    #                     nsamples = input_file.get_event_num_samples(tel_id)
    #                     if nsamples <= 0:
    #                         nsamples = 1
    #                     data.r0.tel[tel_id].num_samples = nsamples
    #  
    #                     # load the data per telescope/pixel
    #                     hessio_mc_npe = input_file.get_mc_number_photon_electron(tel_id)
    #                     data.mc.tel[tel_id].photo_electron_image = hessio_mc_npe
    #                     data.mc.tel[tel_id].meta['refstep'] = (input_file.
    #                                                            get_ref_step(tel_id))
    #                     data.mc.tel[tel_id].time_slice = (input_file.
    #                                                       get_time_slice(tel_id))
    #                     data.mc.tel[tel_id].azimuth_raw = (input_file.
    #                                                        get_azimuth_raw(tel_id))
    #                     data.mc.tel[tel_id].altitude_raw = (input_file.
    #                                                         get_altitude_raw(tel_id))
    #                     data.mc.tel[tel_id].azimuth_cor = (input_file.
    #                                                        get_azimuth_cor(tel_id))
    #                     data.mc.tel[tel_id].altitude_cor = (input_file.
    #                                                         get_altitude_cor(tel_id))

                    elif input_file.attrs['dl_export'] == b"dl1":
                        tels_with_data_tmp[i_tel] = np.bool(input_file['dl1/tels_with_data'][tel_id][i_event])
                        
                        pointing = TelescopePointingContainer()
                        pointing.azimuth = np.deg2rad(input_file['pointing'][tel_id + "_AzCorr"][i_event]) * u.rad
                        pointing.altitude = np.deg2rad(input_file['pointing'][tel_id + "_AltCorr"][i_event]) * u.rad
                        pointing.ra = np.deg2rad(input_file['pointing_radec'][tel_id + "_RaCorr"][i_event]) * u.rad
                        pointing.dec = np.deg2rad(input_file['pointing_radec'][tel_id + "_DecCorr"][i_event]) * u.rad
                        data.pointing[i_tel + 1] = pointing
                        
                        if np.bool(tels_with_data_tmp[i_tel]) == True:
                            
                            data.dl1.tel[i_tel + 1].image = input_file['dl1/tel' + str(i_tel + 1) + '/image'][i_event]
                            data.dl1.tel[i_tel + 1].peakpos = input_file['dl1/tel' + str(i_tel + 1) + '/peakpos'][i_event]
                            data.dl1.tel[i_tel + 1].badpixels = np.array(input_file['dl1/tel' + str(i_tel + 1) + '/badpixels'], dtype=np.bool)
                            
                        if data.meta['is_simulation'] == True:
                            # energy of event should be the same in both telescopes, so simply try both:
                            data.mc.energy = input_file['mc/energy']["Energy"][i_event] * u.TeV
                            data.mc.core_x = input_file['mc/core_xy']["Core_x"][i_event] * u.m
                            data.mc.core_y = input_file['mc/core_xy']["Core_y"][i_event] * u.m
                            data.mc.h_first_int = input_file['mc/h_first_int']["H_first_int"][i_event] * u.m
                        
                # update tels_with_data:
                if tels_with_data_tmp[0] == 1 and tels_with_data_tmp[1] == 0:
                    tels_with_data = {1}
                    time_tmp = Time(input_file['trig/gps_time']["M1_mjd"][i_event], scale='utc', format='mjd') + input_file['trig/gps_time']["M1_sec"][i_event] * u.s
                elif tels_with_data_tmp[0] == 0 and tels_with_data_tmp[1] == 1:
                    tels_with_data = {2}
                    time_tmp = Time(input_file['trig/gps_time']["M2_mjd"][i_event], scale='utc', format='mjd') + input_file['trig/gps_time']["M2_sec"][i_event] * u.s
                elif tels_with_data_tmp[0] == 1 and tels_with_data_tmp[1] == 1:
                    tels_with_data = {1, 2}
                    time_tmp = Time(input_file['trig/gps_time']["M1_mjd"][i_event], scale='utc', format='mjd') + (input_file['trig/gps_time']["M1_sec"][i_event]+input_file['trig/gps_time']["M2_sec"][i_event] )/2. * u.s
                else:
                    tels_with_data = {}
                
                weather = WeatherContainer()
                weather.air_temperature = input_file['weather/air_temperature'][i_event] * u.deg_C
                weather.air_pressure = input_file['weather/air_pressure'][i_event] * u.hPa
                weather.air_humidity = input_file['weather/air_humidity'][i_event] * u.pct
                data.weather = weather
                
                data.trig.gps_time = Time(time_tmp, format='unix', scale='utc', precision=9)
                
                data.r0.tels_with_data = tels_with_data
                data.r1.tels_with_data = tels_with_data
                data.dl0.tels_with_data = tels_with_data
                data.trig.tels_with_trigger = tels_with_data
                data.inst.subarray = magic_subarray
                
                yield data
                counter += 1

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


class MAGICEventSourceROOT(EventSource):
    """
    EventSource for MAGIC calibrated data.

    This class operates with the MAGIC data run-wise. This means that the files
    corresponding to the same data run are loaded and processed together.
    """
    _count = 0

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Constructor

        Parameters
        ----------
        config: traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool: ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs: dict
            Additional parameters to be passed.
            NOTE: The file mask of the data to read can be passed with
            the 'input_url' parameter.
        """

        try:
            import uproot
        except ImportError:
            msg = "The `uproot` python module is required to access the MAGIC data"
            self.log.error(msg)
            raise

        file_list = glob.glob(kwargs['input_url'])
        file_list.sort()

        # EventSource can not handle file wild cards as input_url
        # To overcome this we substitute the input_url with first file matching
        # the specified file mask.
        del kwargs['input_url']
        super().__init__(config=config, tool=tool, input_url=file_list[0], **kwargs)

        # Retrieving the list of run numbers corresponding to the data files
        run_numbers = list(map(self._get_run_number, file_list))
        self.run_numbers = np.unique(run_numbers)

        # # Setting up the current run with the first run present in the data
        # self.current_run = self._set_active_run(run_number=0)
        self.current_run = None
        
        # MAGIC telescope positions in m wrt. to the center of CTA simulations
        self.magic_tel_positions = {
            1: [-27.24, -146.66, 50.00] * u.m,
            2: [-96.44, -96.77, 51.00] * u.m
        }
        # MAGIC telescope description
        optics = OpticsDescription.from_name('MAGIC')
        geom = CameraGeometry.from_name('MAGICCam')
        self.magic_tel_description = TelescopeDescription(optics=optics, camera=geom)
        self.magic_tel_descriptions = {1: self.magic_tel_description, 2: self.magic_tel_description}
        self.magic_subarray = SubarrayDescription('MAGIC', self.magic_tel_positions, self.magic_tel_descriptions)

    @staticmethod
    def is_compatible(file_mask):
        """
        This method checks if the specified file mask corresponds
        to MAGIC data files. The result will be True only if all
        the files are of ROOT format and contain an 'Events' tree.

        Parameters
        ----------
        file_mask: str
            A file mask to check

        Returns
        -------
        bool:
            True if the masked files are MAGIC data runs, False otherwise.

        """

        is_magic_root_file = True

        file_list = glob.glob(file_mask)

        for file_path in file_list:
            try:
                import uproot

                try:
                    with uproot.open(file_path) as input_data:
                        if 'Events' not in input_data:
                            is_magic_root_file = False
                except ValueError:
                    # uproot raises ValueError if the file is not a ROOT file
                    is_magic_root_file = False
                    pass

            except ImportError:
                if re.match('.+_m\d_.+root', file_path.lower()) is None:
                    is_magic_root_file = False

        return is_magic_root_file

    @staticmethod
    def _get_run_number(file_name):
        """
        This internal method extracts the run number from
        the specified file name.

        Parameters
        ----------
        file_name: str
            A file name to process.

        Returns
        -------
        int:
            A run number of the file.
        """

        mask = ".*\d+_M\d+_(\d+)\.\d+_.*"
        parsed_info = re.findall(mask, file_name)

        try:
            run_number = int(parsed_info[0])
        except IndexError:
            raise IndexError('Can not identify the run number of the file {:s}'.format(file_name))

        return run_number

    def _set_active_run(self, run_number):
        """
        This internal method sets the run that will be used for data loading.

        Parameters
        ----------
        run_number: int
            The run number to use.

        Returns
        -------

        """

        input_path = '/'.join(self.input_url.split('/')[:-1])
        this_run_mask = input_path + '/*{:d}*root'.format(run_number)

        run = dict()
        run['number'] = run_number
        run['read_events'] = 0
        run['data'] = MarsDataRun(run_file_mask=this_run_mask)

        return run

    def _generator(self):
        """
        The default event generator. Return the stereo event
        generator instance.

        Returns
        -------

        """

        return self._stereo_event_generator()

    def _stereo_event_generator(self):
        """
        Stereo event generator. Yields DataContainer instances, filled
        with the read event data.

        Returns
        -------

        """

        counter = 0

        # Data container - is initialized once, and data is replaced within it after each yield
        data = DataContainer()
        data.meta['origin'] = "MAGIC"
        data.meta['input_url'] = self.input_url
        data.meta['is_simulation'] = False

        # Telescopes with data:
        tels_in_file = ["m1", "m2"]
        tels_with_data = {1, 2}

        # Loop over the available data runs
        for run_number in self.run_numbers:

            # Removing the previously read data run from memory
            if self.current_run is not None:
                if 'data' in self.current_run:
                    del self.current_run['data']

            # Setting the new active run
            self.current_run = self._set_active_run(run_number)

            # Loop over the events
            for event_i in range(self.current_run['data'].n_stereo_events):
                # Event and run ids
                event_order_number = self.current_run['data'].stereo_ids[event_i][0]
                event_id = self.current_run['data'].event_data['M1']['stereo_event_number'][event_order_number]
                obs_id = self.current_run['number']

                # Reading event data
                event_data = self.current_run['data'].get_stereo_event_data(event_i)

                # Event counter
                data.count = counter

                # Setting up the R0 container
                data.r0.obs_id = obs_id
                data.r0.event_id = event_id
                data.r0.tel.clear()

                # Setting up the R1 container
                data.r1.obs_id = obs_id
                data.r1.event_id = event_id
                data.r1.tel.clear()

                # Setting up the DL0 container
                data.dl0.obs_id = obs_id
                data.dl0.event_id = event_id
                data.dl0.tel.clear()

                # Filling the DL1 container with the event data
                for tel_i, tel_id in enumerate(tels_in_file):
                    # Creating the telescope pointing container
                    pointing = TelescopePointingContainer()
                    pointing.azimuth = np.deg2rad(event_data['pointing_az']) * u.rad
                    pointing.altitude = np.deg2rad(90 - event_data['pointing_zd']) * u.rad
                    pointing.ra = np.deg2rad(event_data['pointing_ra']) * u.rad
                    pointing.dec = np.deg2rad(event_data['pointing_dec']) * u.rad

                    # Adding the pointing container to the event data
                    data.pointing[tel_i + 1] = pointing

                    # Adding event charge and peak positions per pixel
                    data.dl1.tel[tel_i + 1].image = event_data['{:s}_image'.format(tel_id)]
                    data.dl1.tel[tel_i + 1].peakpos = event_data['{:s}_peak_pos'.format(tel_id)]
                    # data.dl1.tel[i_tel + 1].badpixels = np.array(
                    #     file['dl1/tel' + str(i_tel + 1) + '/badpixels'], dtype=np.bool)

                # Adding the event arrival time
                time_tmp = Time(event_data['mjd'], scale='utc', format='mjd')
                data.trig.gps_time = Time(time_tmp, format='unix', scale='utc', precision=9)

                # Setting the telescopes with data
                data.r0.tels_with_data = tels_with_data
                data.r1.tels_with_data = tels_with_data
                data.dl0.tels_with_data = tels_with_data
                data.trig.tels_with_trigger = tels_with_data

                # Setting the instrument sub-array
                data.inst.subarray = self.magic_subarray

                yield data
                counter += 1

        return

    def _mono_event_generator(self, telescope):
        """
        Mono event generator. Yields DataContainer instances, filled
        with the read event data.

        Parameters
        ----------
        telescope: str
            The telescope for which to return events. Can be either "M1" or "M2".

        Returns
        -------

        """

        counter = 0
        telescope = telescope.upper()

        # Data container - is initialized once, and data is replaced within it after each yield
        data = DataContainer()
        data.meta['origin'] = "MAGIC"
        data.meta['input_url'] = self.input_url
        data.meta['is_simulation'] = False

        # Telescopes with data:
        tels_in_file = ["M1", "M2"]

        if telescope not in tels_in_file:
            raise ValueError("Specified telescope {:s} is not in the allowed list {}".format(telescope, tels_in_file))

        tel_i = tels_in_file.index(telescope)
        tels_with_data = {tel_i + 1, }

        # Loop over the available data runs
        for run_number in self.run_numbers:

            # Removing the previously read data run from memory
            if self.current_run is not None:
                if 'data' in self.current_run:
                    del self.current_run['data']

            # Setting the new active run
            self.current_run = self._set_active_run(run_number)

            if telescope == 'M1':
                n_events = self.current_run['data'].n_mono_events_m1
            else:
                n_events = self.current_run['data'].n_mono_events_m2

            # Loop over the events
            for event_i in range(n_events):
                # Event and run ids
                event_order_number = self.current_run['data'].mono_ids[telescope][event_i]
                event_id = self.current_run['data'].event_data[telescope]['stereo_event_number'][event_order_number]
                obs_id = self.current_run['number']

                # Reading event data
                event_data = self.current_run['data'].get_mono_event_data(event_i, telescope=telescope)

                # Event counter
                data.count = counter

                # Setting up the R0 container
                data.r0.obs_id = obs_id
                data.r0.event_id = event_id
                data.r0.tel.clear()

                # Setting up the R1 container
                data.r1.obs_id = obs_id
                data.r1.event_id = event_id
                data.r1.tel.clear()

                # Setting up the DL0 container
                data.dl0.obs_id = obs_id
                data.dl0.event_id = event_id
                data.dl0.tel.clear()

                # Creating the telescope pointing container
                pointing = TelescopePointingContainer()
                pointing.azimuth = np.deg2rad(event_data['pointing_az']) * u.rad
                pointing.altitude = np.deg2rad(90 - event_data['pointing_zd']) * u.rad
                pointing.ra = np.deg2rad(event_data['pointing_ra']) * u.rad
                pointing.dec = np.deg2rad(event_data['pointing_dec']) * u.rad

                # Adding the pointing container to the event data
                data.pointing[tel_i + 1] = pointing

                # Adding event charge and peak positions per pixel
                data.dl1.tel[tel_i + 1].image = event_data['image']
                data.dl1.tel[tel_i + 1].peakpos = event_data['peak_pos']
                # data.dl1.tel[tel_i + 1].badpixels = np.array(
                #     file['dl1/tel' + str(i_tel + 1) + '/badpixels'], dtype=np.bool)

                # Adding the event arrival time
                time_tmp = Time(event_data['mjd'], scale='utc', format='mjd')
                data.trig.gps_time = Time(time_tmp, format='unix', scale='utc', precision=9)

                # Setting the telescopes with data
                data.r0.tels_with_data = tels_with_data
                data.r1.tels_with_data = tels_with_data
                data.dl0.tels_with_data = tels_with_data
                data.trig.tels_with_trigger = tels_with_data

                # Setting the instrument sub-array
                data.inst.subarray = self.magic_subarray

                yield data
                counter += 1

        return

    def _pedestal_event_generator(self, telescope):
        """
        Pedestal event generator. Yields DataContainer instances, filled
        with the read event data.

        Parameters
        ----------
        telescope: str
            The telescope for which to return events. Can be either "M1" or "M2".

        Returns
        -------

        """

        counter = 0
        telescope = telescope.upper()

        # Data container - is initialized once, and data is replaced within it after each yield
        data = DataContainer()
        data.meta['origin'] = "MAGIC"
        data.meta['input_url'] = self.input_url
        data.meta['is_simulation'] = False

        # Telescopes with data:
        tels_in_file = ["M1", "M2"]

        if telescope not in tels_in_file:
            raise ValueError("Specified telescope {:s} is not in the allowed list {}".format(telescope, tels_in_file))

        tel_i = tels_in_file.index(telescope)
        tels_with_data = {tel_i + 1, }

        # Loop over the available data runs
        for run_number in self.run_numbers:

            # Removing the previously read data run from memory
            if self.current_run is not None:
                if 'data' in self.current_run:
                    del self.current_run['data']

            # Setting the new active run
            self.current_run = self._set_active_run(run_number)

            if telescope == 'M1':
                n_events = self.current_run['data'].n_pedestal_events_m1
            else:
                n_events = self.current_run['data'].n_pedestal_events_m2

            # Loop over the events
            for event_i in range(n_events):
                # Event and run ids
                event_order_number = self.current_run['data'].pedestal_ids[telescope][event_i]
                event_id = self.current_run['data'].event_data[telescope]['stereo_event_number'][event_order_number]
                obs_id = self.current_run['number']

                # Reading event data
                event_data = self.current_run['data'].get_pedestal_event_data(event_i, telescope=telescope)

                # Event counter
                data.count = counter

                # Setting up the R0 container
                data.r0.obs_id = obs_id
                data.r0.event_id = event_id
                data.r0.tel.clear()

                # Setting up the R1 container
                data.r1.obs_id = obs_id
                data.r1.event_id = event_id
                data.r1.tel.clear()

                # Setting up the DL0 container
                data.dl0.obs_id = obs_id
                data.dl0.event_id = event_id
                data.dl0.tel.clear()

                # Creating the telescope pointing container
                pointing = TelescopePointingContainer()
                pointing.azimuth = np.deg2rad(event_data['pointing_az']) * u.rad
                pointing.altitude = np.deg2rad(90 - event_data['pointing_zd']) * u.rad
                pointing.ra = np.deg2rad(event_data['pointing_ra']) * u.rad
                pointing.dec = np.deg2rad(event_data['pointing_dec']) * u.rad

                # Adding the pointing container to the event data
                data.pointing[tel_i + 1] = pointing

                # Adding event charge and peak positions per pixel
                data.dl1.tel[tel_i + 1].image = event_data['image']
                data.dl1.tel[tel_i + 1].peakpos = event_data['peak_pos']
                # data.dl1.tel[tel_i + 1].badpixels = np.array(
                #     file['dl1/tel' + str(i_tel + 1) + '/badpixels'], dtype=np.bool)

                # Adding the event arrival time
                time_tmp = Time(event_data['mjd'], scale='utc', format='mjd')
                data.trig.gps_time = Time(time_tmp, format='unix', scale='utc', precision=9)

                # Setting the telescopes with data
                data.r0.tels_with_data = tels_with_data
                data.r1.tels_with_data = tels_with_data
                data.dl0.tels_with_data = tels_with_data
                data.trig.tels_with_trigger = tels_with_data

                # Setting the instrument sub-array
                data.inst.subarray = self.magic_subarray

                yield data
                counter += 1

        return


class MarsDataRun:
    """
    This class implements reading of the event data from a single MAGIC data run.
    """

    def __init__(self, run_file_mask):
        """
        Constructor of the class. Defines the run to use and the camera pixel arrangement.

        Parameters
        ----------
        run_file_mask: str
            A path mask for files belonging to the run. Must correspond to a single run
            or an exception will be raised. Must correspond to calibrated ("sorcerer"-level)
            data.
        """

        self.run_file_mask = run_file_mask

        # Loading the camera geometry
        camera_geometry = CameraGeometry.from_name('MAGICCam')
        self.camera_pixel_x = camera_geometry.pix_x.value
        self.camera_pixel_y = camera_geometry.pix_y.value
        self.n_camera_pixels = len(self.camera_pixel_x)

        # Preparing the lists of M1/2 data files
        file_list = glob.glob(run_file_mask)
        self.m1_file_list = list(filter(lambda name: '_M1_' in name, file_list))
        self.m1_file_list.sort()
        self.m2_file_list = list(filter(lambda name: '_M2_' in name, file_list))
        self.m2_file_list.sort()

        # Retrieving the list of run numbers corresponding to the data files
        run_numbers = list(map(self._get_run_number, file_list))
        run_numbers = np.unique(run_numbers)

        # Checking if a single run is going to be read
        if len(run_numbers) > 1:
            raise ValueError("Run mask corresponds to more than one run: {}".format(run_numbers))

        # Reading the event data
        self.event_data = dict()
        self.event_data['M1'] = self.load_events(self.m1_file_list)
        self.event_data['M2'] = self.load_events(self.m2_file_list)

        # Detecting pedestal events
        self.pedestal_ids = self._find_pedestal_events()
        # Detecting stereo events
        self.stereo_ids = self._find_stereo_events()
        # Detecting mono events
        self.mono_ids = self._find_mono_events()

    @property
    def n_events_m1(self):
        return len(self.event_data['M1']['MJD'])

    @property
    def n_events_m2(self):
        return len(self.event_data['M2']['MJD'])

    @property
    def n_stereo_events(self):
        return len(self.stereo_ids)

    @property
    def n_mono_events_m1(self):
        return len(self.mono_ids['M1'])

    @property
    def n_mono_events_m2(self):
        return len(self.mono_ids['M2'])

    @property
    def n_pedestal_events_m1(self):
        return len(self.pedestal_ids['M1'])

    @property
    def n_pedestal_events_m2(self):
        return len(self.pedestal_ids['M2'])

    @staticmethod
    def _get_run_number(file_name):
        """
        This internal method extracts the run number from
        a specified file name.

        Parameters
        ----------
        file_name: str
            A file name to process.

        Returns
        -------
        int:
            A run number of the file.
        """

        mask = ".*\d+_M\d+_(\d+)\.\d+_.*"
        parsed_info = re.findall(mask, file_name)

        run_number = int(parsed_info[0])

        return run_number

    @staticmethod
    def load_events(file_list):
        """
        This method loads events from the pre-defiled file and returns them as a dictionary.

        Parameters
        ----------
        file_name: str
            Name of the MAGIC calibrated file to use.

        Returns
        -------
        dict:
            A dictionary with the even properties: charge / arrival time data, trigger, direction etc.
        """

        try:
            import uproot
        except ImportError:
            msg = "The `uproot` python module is required to access the MAGIC data"
            raise ImportError(msg)

        event_data = dict()

        event_data['charge'] = []
        event_data['arrival_time'] = []
        event_data['trigger_pattern'] = np.array([])
        event_data['stereo_event_number'] = np.array([])
        event_data['pointing_zd'] = np.array([])
        event_data['pointing_az'] = np.array([])
        event_data['pointing_ra'] = np.array([])
        event_data['pointing_dec'] = np.array([])
        event_data['MJD'] = np.array([])

        event_data['file_edges'] = [0]

        for file_name in file_list:

            input_file = uproot.open(file_name)

            # Reading the info common to MC and real data
            charge = input_file['Events']['MCerPhotEvt.fPixels.fPhot'].array()
            arrival_time = input_file['Events']['MArrivalTime.fData'].array()
            trigger_pattern = input_file['Events']['MTriggerPattern.fPrescaled'].array()
            stereo_event_number = input_file['Events']['MRawEvtHeader.fStereoEvtNumber'].array()

            # Computing the event arrival time
            mjd = input_file['Events']['MTime.fMjd'].array()
            millisec = input_file['Events']['MTime.fTime.fMilliSec'].array()
            nanosec = input_file['Events']['MTime.fNanoSec'].array()

            mjd = mjd + (millisec/1e3 + nanosec/1e9) / 86400.0

            degrees_per_hour = 15.0

            if 'MPointingPos.' in input_file['Events']:
                # Retrieving the telescope pointing direction
                pointing_zd = input_file['Events']['MPointingPos.fZd'].array() - input_file['Events']['MPointingPos.fDevZd'].array()
                pointing_az = input_file['Events']['MPointingPos.fAz'].array() - input_file['Events']['MPointingPos.fDevAz'].array()
                pointing_ra = (input_file['Events']['MPointingPos.fRa'].array() - input_file['Events']['MPointingPos.fDevHa'].array()) * degrees_per_hour
                pointing_dec = input_file['Events']['MPointingPos.fDec'].array() - input_file['Events']['MPointingPos.fDevDec'].array()
            else:
                # Getting the telescope drive info
                drive_mjd = input_file['Drive']['MReportDrive.fMjd'].array()
                drive_zd = input_file['Drive']['MReportDrive.fCurrentZd'].array()
                drive_az = input_file['Drive']['MReportDrive.fCurrentAz'].array()
                drive_ra = input_file['Drive']['MReportDrive.fRa'].array() * degrees_per_hour
                drive_dec = input_file['Drive']['MReportDrive.fDec'].array()

                # Creating azimuth and zenith angles interpolators
                drive_zd_pointing_interpolator = scipy.interpolate.interp1d(drive_mjd, drive_zd, fill_value="extrapolate")
                drive_az_pointing_interpolator = scipy.interpolate.interp1d(drive_mjd, drive_az, fill_value="extrapolate")

                # Creating azimuth and zenith angles interpolators
                drive_ra_pointing_interpolator = scipy.interpolate.interp1d(drive_mjd, drive_ra, fill_value="extrapolate")
                drive_dec_pointing_interpolator = scipy.interpolate.interp1d(drive_mjd, drive_dec, fill_value="extrapolate")

                # Interpolating the drive pointing to the event time stamps
                pointing_zd = drive_zd_pointing_interpolator(event_data['MJD'])
                pointing_az = drive_az_pointing_interpolator(event_data['MJD'])
                pointing_ra = drive_ra_pointing_interpolator(event_data['MJD'])
                pointing_dec = drive_dec_pointing_interpolator(event_data['MJD'])

            event_data['charge'].append(charge)
            event_data['arrival_time'].append(arrival_time)
            event_data['trigger_pattern'] = np.concatenate((event_data['trigger_pattern'], trigger_pattern))
            event_data['stereo_event_number'] = np.concatenate((event_data['stereo_event_number'], stereo_event_number))
            event_data['pointing_zd'] = np.concatenate((event_data['pointing_zd'], pointing_zd))
            event_data['pointing_az'] = np.concatenate((event_data['pointing_az'], pointing_az))
            event_data['pointing_ra'] = np.concatenate((event_data['pointing_ra'], pointing_ra))
            event_data['pointing_dec'] = np.concatenate((event_data['pointing_dec'], pointing_dec))

            event_data['MJD'] = np.concatenate((event_data['MJD'], mjd))

            event_data['file_edges'].append(len(event_data['trigger_pattern']))

        return event_data

    def _find_pedestal_events(self):
        """
        This internal method identifies the IDs (order numbers) of the
        pedestal events in the run.

        Returns
        -------
        dict:
            A dictionary of pedestal event IDs in M1/2 separately.
        """

        pedestal_ids = dict()

        pedestal_trigger_pattern = 8

        for telescope in self.event_data:
            ped_triggers = np.where(self.event_data[telescope]['trigger_pattern'] == pedestal_trigger_pattern)
            pedestal_ids[telescope] = ped_triggers[0]

        return pedestal_ids

    def _find_stereo_events(self):
        """
        This internal methods identifies stereo events in the run.

        Returns
        -------
        list:
            A list of pairs (M1_id, M2_id) corresponding to stereo events in the run.
        """

        data_trigger_pattern = 128

        m2_data_condition = (self.event_data['M2']['trigger_pattern'] == data_trigger_pattern)

        stereo_ids = []
        n_m1_events = len(self.event_data['M1']['stereo_event_number'])

        for m1_id in range(0, n_m1_events):
            if self.event_data['M1']['trigger_pattern'][m1_id] == data_trigger_pattern:
                m2_stereo_condition = (self.event_data['M2']['stereo_event_number'] ==
                                       self.event_data['M1']['stereo_event_number'][m1_id])

                m12_match = np.where(m2_data_condition & m2_stereo_condition)

                if len(m12_match[0]) > 0:
                    stereo_pair = (m1_id, m12_match[0][0])
                    stereo_ids.append(stereo_pair)

        return stereo_ids

    def _find_mono_events(self):
        """
        This internal method identifies the IDs (order numbers) of the
        pedestal events in the run.

        Returns
        -------
        dict:
            A dictionary of pedestal event IDs in M1/2 separately.
        """

        mono_ids = dict()
        mono_ids['M1'] = []
        mono_ids['M2'] = []

        data_trigger_pattern = 128

        m1_data_condition = self.event_data['M1']['trigger_pattern'] == data_trigger_pattern
        m2_data_condition = self.event_data['M2']['trigger_pattern'] == data_trigger_pattern

        n_m1_events = len(self.event_data['M1']['stereo_event_number'])
        n_m2_events = len(self.event_data['M2']['stereo_event_number'])

        for m1_id in range(0, n_m1_events):
            if m1_data_condition[m1_id]:
                m2_stereo_condition = (self.event_data['M2']['stereo_event_number'] ==
                                       self.event_data['M1']['stereo_event_number'][m1_id])

                m12_match = np.where(m2_data_condition & m2_stereo_condition)

                if len(m12_match[0]) == 0:
                    mono_ids['M1'].append(m1_id)

        for m2_id in range(0, n_m2_events):
            if m2_data_condition[m2_id]:
                m1_stereo_condition = (self.event_data['M1']['stereo_event_number'] ==
                                       self.event_data['M2']['stereo_event_number'][m2_id])

                m12_match = np.where(m1_data_condition & m1_stereo_condition)

                if len(m12_match[0]) == 0:
                    mono_ids['M2'].append(m2_id)

        return mono_ids

    def _get_pedestal_file_num(self, pedestal_event_num, telescope):
        """
        This internal method identifies the M1/2 file number of the
        given pedestal event in M1/2 file lists, corresponding to this run.

        Parameters
        ----------
        pedestal_event_num: int
            Order number of the event in the list of pedestal events
            of the specified telescope, corresponding to this run.
        telescope: str
            The name of the telescope to which this event corresponds.
            May be "M1" or "M2".

        Returns
        -------
        file_num:
            Order number of the corresponding file in the M1 or M2 file list.
        """

        event_id = self.pedestal_ids[telescope][pedestal_event_num]
        file_num = np.digitize([event_id], self.event_data[telescope]['file_edges'])
        file_num = file_num[0] - 1

        return file_num

    def _get_stereo_file_num(self, stereo_event_num):
        """
        This internal method identifies the M1/2 file numbers of the
        given stereo event in M1/2 file lists, corresponding to this run.

        Parameters
        ----------
        stereo_event_num: int
            Order number of the event in the list of stereo events corresponding
            to this run.

        Returns
        -------
        m1_file_num:
            Order number of the corresponding file in the M1 file list.
        m2_file_num:
            Order number of the corresponding file in the M2 file list.
        """

        m1_id = self.stereo_ids[stereo_event_num][0]
        m2_id = self.stereo_ids[stereo_event_num][1]
        m1_file_num = np.digitize([m1_id], self.event_data['M1']['file_edges'])
        m2_file_num = np.digitize([m2_id], self.event_data['M2']['file_edges'])

        m1_file_num = m1_file_num[0] - 1
        m2_file_num = m2_file_num[0] - 1

        return m1_file_num, m2_file_num

    def _get_mono_file_num(self, mono_event_num, telescope):
        """
        This internal method identifies the M1/2 file number of the
        given mono event in M1/2 file lists, corresponding to this run.

        Parameters
        ----------
        mono_event_num: int
            Order number of the event in the list of stereo events corresponding
            to this run.
        telescope: str
            The name of the telescope to which this event corresponds.
            May be "M1" or "M2".

        Returns
        -------
        file_num:
            Order number of the corresponding file in the M1 or M2 file list.
        """

        event_id = self.mono_ids[telescope][mono_event_num]
        file_num = np.digitize([event_id], self.event_data[telescope]['file_edges'])
        file_num = file_num[0] - 1

        return file_num

    def get_pedestal_event_data(self, pedestal_event_num, telescope):
        """
        This method read the photon content and arrival time (per pixel)
        for the specified pedestal event. Also returned is the event telescope pointing
        data.

        Parameters
        ----------
        pedestal_event_num: int
            Order number of the event in the list of pedestal events for the
            given telescope, corresponding to this run.
        telescope: str
            The name of the telescope to which this event corresponds.
            May be "M1" or "M2".

        Returns
        -------
        dict:
            The output has the following structure:
            'image' - photon_content in requested telescope
            'peak_pos' - arrival_times in requested telescope
            'pointing_az' - pointing azimuth
            'pointing_zd' - pointing zenith angle
            'pointing_ra' - pointing right ascension
            'pointing_dec' - pointing declination
            'mjd' - event arrival time
        """

        file_num = self._get_pedestal_file_num(pedestal_event_num, telescope)
        event_id = self.pedestal_ids[telescope][pedestal_event_num]

        id_in_file = event_id - self.event_data[telescope]['file_edges'][file_num]

        photon_content = self.event_data[telescope]['charge'][file_num][id_in_file][:self.n_camera_pixels]
        arrival_times = self.event_data[telescope]['arrival_time'][file_num][id_in_file][:self.n_camera_pixels]

        event_data = dict()
        event_data['image'] = photon_content
        event_data['peak_pos'] = arrival_times
        event_data['pointing_az'] = self.event_data[telescope]['pointing_az'][event_id]
        event_data['pointing_zd'] = self.event_data[telescope]['pointing_zd'][event_id]
        event_data['pointing_ra'] = self.event_data[telescope]['pointing_ra'][event_id]
        event_data['pointing_dec'] = self.event_data[telescope]['pointing_dec'][event_id]
        event_data['mjd'] = self.event_data[telescope]['MJD'][event_id]

        return event_data

    def get_stereo_event_data(self, stereo_event_num):
        """
        This method read the photon content and arrival time (per pixel)
        for the specified stereo event. Also returned is the event telescope pointing
        data.

        Parameters
        ----------
        stereo_event_num: int
            Order number of the event in the list of stereo events corresponding
            to this run.

        Returns
        -------
        dict:
            The output has the following structure:
            'm1_image' - M1 photon_content
            'm1_peak_pos' - M1 arrival_times
            'm2_image' - M2 photon_content
            'm2_peak_pos' - M2 arrival_times
            'pointing_az' - pointing azimuth
            'pointing_zd' - pointing zenith angle
            'pointing_ra' - pointing right ascension
            'pointing_dec' - pointing declination
            'mjd' - event arrival time
        """

        m1_file_num, m2_file_num = self._get_stereo_file_num(stereo_event_num)
        m1_id = self.stereo_ids[stereo_event_num][0]
        m2_id = self.stereo_ids[stereo_event_num][1]

        m1_id_in_file = m1_id - self.event_data['M1']['file_edges'][m1_file_num]
        m2_id_in_file = m2_id - self.event_data['M2']['file_edges'][m2_file_num]

        m1_photon_content = self.event_data['M1']['charge'][m1_file_num][m1_id_in_file][:self.n_camera_pixels]
        m1_arrival_times = self.event_data['M1']['arrival_time'][m1_file_num][m1_id_in_file][:self.n_camera_pixels]

        m2_photon_content = self.event_data['M2']['charge'][m2_file_num][m2_id_in_file][:self.n_camera_pixels]
        m2_arrival_times = self.event_data['M2']['arrival_time'][m2_file_num][m2_id_in_file][:self.n_camera_pixels]

        event_data = dict()
        event_data['m1_image'] = m1_photon_content
        event_data['m1_peak_pos'] = m1_arrival_times
        event_data['m2_image'] = m2_photon_content
        event_data['m2_peak_pos'] = m2_arrival_times
        event_data['pointing_az'] = self.event_data['M1']['pointing_az'][m1_id]
        event_data['pointing_zd'] = self.event_data['M1']['pointing_zd'][m1_id]
        event_data['pointing_ra'] = self.event_data['M1']['pointing_ra'][m1_id]
        event_data['pointing_dec'] = self.event_data['M1']['pointing_dec'][m1_id]
        event_data['mjd'] = self.event_data['M1']['MJD'][m1_id]

        return event_data

    def get_mono_event_data(self, mono_event_num, telescope):
        """
        This method read the photon content and arrival time (per pixel)
        for the specified mono event. Also returned is the event telescope pointing
        data.

        Parameters
        ----------
        mono_event_num: int
            Order number of the event in the list of mono events for the
            given telescope, corresponding to this run.
        telescope: str
            The name of the telescope to which this event corresponds.
            May be "M1" or "M2".

        Returns
        -------
        dict:
            The output has the following structure:
            'image' - photon_content in requested telescope
            'peak_pos' - arrival_times in requested telescope
            'pointing_az' - pointing azimuth
            'pointing_zd' - pointing zenith angle
            'pointing_ra' - pointing right ascension
            'pointing_dec' - pointing declination
            'mjd' - event arrival time
        """

        file_num = self._get_mono_file_num(mono_event_num, telescope)
        event_id = self.mono_ids[telescope][mono_event_num]

        id_in_file = event_id - self.event_data[telescope]['file_edges'][file_num]

        photon_content = self.event_data[telescope]['charge'][file_num][id_in_file][:self.n_camera_pixels]
        arrival_times = self.event_data[telescope]['arrival_time'][file_num][id_in_file][:self.n_camera_pixels]

        event_data = dict()
        event_data['image'] = photon_content
        event_data['peak_pos'] = arrival_times
        event_data['pointing_az'] = self.event_data[telescope]['pointing_az'][event_id]
        event_data['pointing_zd'] = self.event_data[telescope]['pointing_zd'][event_id]
        event_data['pointing_ra'] = self.event_data[telescope]['pointing_ra'][event_id]
        event_data['pointing_dec'] = self.event_data[telescope]['pointing_dec'][event_id]
        event_data['mjd'] = self.event_data[telescope]['MJD'][event_id]

        return event_data
