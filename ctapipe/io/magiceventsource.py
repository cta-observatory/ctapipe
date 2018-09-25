
import numpy as np

from astropy import units as u
from astropy.units import cds
cds.enable()  
from astropy.coordinates import Angle
from astropy.time import Time
from ctapipe.io.eventsource import EventSource
from ctapipe.io.containers import DataContainer, TelescopePointingContainer
from ctapipe.instrument import TelescopeDescription, SubarrayDescription, OpticsDescription, CameraGeometry
import gzip
import struct

__all__ = ['MAGICEventSource']


class MAGICEventSource(EventSource):
    """
    EventSource for MAGIC raw data converted to hdf5.

    This class utilises `h5py` to read the hdf5 file, and stores the
    information into the event containers.
    """
    _count = 0

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

        try:
            import h5py
        except ImportError:
            msg = "The `h5py` python module is required to access the MAGIC data or MCs"
            self.log.error(msg)
            raise

        self.h5py = h5py
        self.file = h5py.File(self.input_url)

#         if MAGICEventSource._count > 0:
#             self.log.warn("Only one MAGIC raw event_source allowed at a time. "
#                           "Previous hdf5 file will be closed.")
#             self.pyhessio.close_file()
#         MAGICEventSource._count += 1


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
        with self.h5py.File(self.input_url, "r") as file:
            # the container is initialized once, and data is replaced within
            # it after each yield
            counter = 0
            #eventstream = file.move_to_next_event()
            data = DataContainer()
            data.meta['origin'] = "MAGIC"

            # some hessio_event_source specific parameters
            data.meta['input_url'] = self.input_url
            data.meta['max_events'] = self.max_events

            events_per_tel = [[],[]]

            if not file.attrs['RunType'] == "Data":
                data.meta['is_simulation'] = True
                dt = np.float
            else:
                data.meta['is_simulation'] = False
                dt = np.int16
                
            # check which telescopes have data:
            
            if file.attrs['dl_export'] == b"dl1":
                eventstream = file['dl1/event_id']
                if file.attrs['data format'] == b"mono M1":
                    tels_in_file = ["M1"]
                elif file.attrs['data format'] == b"mono M2":
                    tels_in_file = ["M2"]
                elif file.attrs['data format'] == b"stereo":
                    tels_in_file = ["M1", "M2"]
                
            elif file.attrs['dl_export'] == b"r0":
                if list(file.keys()) == ["MAGIC1_rawdata"]:
                    eventstream = file['MAGIC1_rawdata/EvtHeader/StereoEvtNumber']
                    events_per_tel[0] = np.array(eventstream, dtype = dt)
                    tels_in_file = [1]
                elif list(file.keys()) == ["MAGIC2_rawdata"]:
                    eventstream = file['MAGIC2_rawdata/EvtHeader/StereoEvtNumber']
                    events_per_tel[1] = np.array(eventstream, dtype = dt)
                    tels_in_file = [2]
                elif list(file.keys()) == ["MAGIC1_rawdata", "MAGIC2_rawdata"]:
                    events_per_tel[0] = np.array(file['MAGIC1_rawdata/EvtHeader/StereoEvtNumber'], dtype = dt)
                    events_per_tel[1] = np.array(file['MAGIC2_rawdata/EvtHeader/StereoEvtNumber'], dtype = dt)
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
                raise IOError("MAGIC data level attribute 'dl_export' not found in file (value should be 'r0' or 'dl1').")
            
            optics = OpticsDescription.from_name(str(file['inst/subarray'].attrs['OpticsDescription'])[2:-1])
            geom = CameraGeometry.from_name(str(file['inst/subarray'].attrs['CameraGeometry'])[2:-1])
            magic_tel_description = TelescopeDescription(optics=optics, camera=geom)
            magic_tel_descriptions = {1: magic_tel_description, 2: magic_tel_description}
            magic_tel_positions = {1: [file['inst/subarray/tel_coords']['M1'][0]*u.m, file['inst/subarray/tel_coords']['M1'][1]*u.m, file['inst/subarray/tel_coords']['M1'][2]*u.m], 
                                   2: [file['inst/subarray/tel_coords']['M2'][0]*u.m, file['inst/subarray/tel_coords']['M2'][1]*u.m, file['inst/subarray/tel_coords']['M2'][2]*u.m]}
            magic_subarray = SubarrayDescription(str(file.attrs['instrument'])[2:-1], magic_tel_positions, magic_tel_descriptions)

            
            for i_event, event_id in enumerate(eventstream):

#                 if counter == 0:
#                     # subarray info is only available when an event is loaded,
#                     # so load it on the first event.
#                     data.inst.subarray = self._build_subarray_info(file)
# 
                obs_id = file.attrs['RunNumber']
                data.count = counter
                data.r0.obs_id = file.attrs['RunNumber']
                data.r0.event_id = event_id
                data.r1.obs_id = file.attrs['RunNumber']
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
#                 data.trig.tels_with_trigger = (file.
#                                                get_central_event_teltrg_list())
#                 time_s, time_ns = file.get_central_event_gps_time()
#                 data.trig.gps_time = Time(time_s * u.s, time_ns * u.ns,
#                                           format='unix', scale='utc')

#                 data.mc.alt = Angle(file.get_mc_shower_altitude(), u.rad)
#                 data.mc.az = Angle(file.get_mc_shower_azimuth(), u.rad)

#                 data.mc.x_max = file.get_mc_shower_xmax() * u.g / (u.cm**2)
#                 data.mc.shower_primary_id = file.get_mc_shower_primary_id()
# 
#                 # mc run header data
#                 data.mcheader.run_array_direction = Angle(
#                     file.get_mc_run_array_direction() * u.rad
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
                    if file.attrs['dl_export'] == b"r0":
                        # search event
                        if event_id not in events_per_tel[i_tel]:
                            nevent = -1
                        else:
                            nevent = np.searchsorted(events_per_tel[i_tel], event_id, side='left')
                            tels_with_data_tmp[i_tel] = 1
                            
                        # sort out remaining calibration runs:
                        if file['MAGIC'+str(tel_id) +'_rawdata/EvtHeader/TrigPattern']["L3 trigger"][nevent] == False:
                            nevent = -1
                            tels_with_data_tmp[i_tel] = 0
    #                     # event.mc.tel[tel_id] = MCCameraContainer()
    # 
    #                     data.mc.tel[tel_id].dc_to_pe = file.get_calibration(tel_id)
    #                     data.mc.tel[tel_id].pedestal = file.get_pedestal(tel_id)
                        if nevent == -1:
                            data.r0.tel[tel_id].waveform = None
                        else:
                            data.r0.tel[tel_id].waveform = file['MAGIC'+str(tel_id) +'_rawdata/Events'][...,nevent]
    
                        data.r0.tel[tel_id].image = np.sum(data.r0.tel[tel_id].waveform, axis=0)
                        data.r0.tel[tel_id].num_trig_pix = file['MAGIC'+str(tel_id) +'_rawdata/EvtHeader/NumTrigLvl2'][nevent]
                        
                        # add MC information:
                        if data.meta['is_simulation'] == True:
                            # energy of event should be the same in both telescopes, so simply try both:
                            data.mc.energy = file['MAGIC'+str(tel_id) +'_rawdata/McHeader/Energy']["Energy"][nevent] * u.TeV
                            data.mc.core_x = file['MAGIC'+str(tel_id) +'_rawdata/McHeader/Core_xy']["Core_x"][nevent] * u.m
                            data.mc.core_y = file['MAGIC'+str(tel_id) +'_rawdata/McHeader/Core_xy']["Core_y"][nevent] * u.m
                            data.mc.h_first_int = file['MAGIC'+str(tel_id) +'_rawdata/McHeader/H_first_int']["H_first_int"][nevent] * u.m
                        
    #                     data.r0.tel[tel_id].trig_pix_id = file.get_trig_pixels(tel_id)
    #                     data.mc.tel[tel_id].reference_pulse_shape = (file.
    #                                                                  get_ref_shapes(tel_id))
    #  
    #                     nsamples = file.get_event_num_samples(tel_id)
    #                     if nsamples <= 0:
    #                         nsamples = 1
    #                     data.r0.tel[tel_id].num_samples = nsamples
    #  
    #                     # load the data per telescope/pixel
    #                     hessio_mc_npe = file.get_mc_number_photon_electron(tel_id)
    #                     data.mc.tel[tel_id].photo_electron_image = hessio_mc_npe
    #                     data.mc.tel[tel_id].meta['refstep'] = (file.
    #                                                            get_ref_step(tel_id))
    #                     data.mc.tel[tel_id].time_slice = (file.
    #                                                       get_time_slice(tel_id))
    #                     data.mc.tel[tel_id].azimuth_raw = (file.
    #                                                        get_azimuth_raw(tel_id))
    #                     data.mc.tel[tel_id].altitude_raw = (file.
    #                                                         get_altitude_raw(tel_id))
    #                     data.mc.tel[tel_id].azimuth_cor = (file.
    #                                                        get_azimuth_cor(tel_id))
    #                     data.mc.tel[tel_id].altitude_cor = (file.
    #                                                         get_altitude_cor(tel_id))

                    elif file.attrs['dl_export'] == b"dl1":
                        tels_with_data_tmp[i_tel] = file['dl1/tels_with_data'][tel_id][i_event]
                        
                        if tels_with_data_tmp[i_tel] == 1:
                            pointing = TelescopePointingContainer()
                            pointing.azimuth = np.deg2rad(file['pointing'][tel_id + "_AzCorr"][i_event]) * u.rad
                            pointing.altitude = np.deg2rad(file['pointing'][tel_id + "_DecCorr"][i_event]) * u.rad
                            data.pointing[i_tel + 1] = pointing
                            
                            time = Time(file['trig/gps_time'][tel_id + "_mjd"][i_event] * cds.MJD, file['trig/gps_time'][tel_id + "_sec"][i_event] * u.s,
                                      format='unix', scale='utc', precision=9)
                            
                            data.dl1.tel[i_tel + 1].image = file['dl1/tel' + str(i_tel + 1) + '/image'][i_event]
                            data.dl1.tel[i_tel + 1].peakpos = file['dl1/tel' + str(i_tel + 1) + '/peakpos'][i_event]
                            data.dl1.tel[i_tel + 1].badpixels = np.array(file['dl1/tel' + str(i_tel + 1) + '/badpixels'], dtype=np.bool)
                            
                            
                        if data.meta['is_simulation'] == True:
                            # energy of event should be the same in both telescopes, so simply try both:
                            data.mc.energy = file['mc/energy']["Energy"][i_event] * u.TeV
                            data.mc.core_x = file['mc/core_xy']["Core_x"][i_event] * u.m
                            data.mc.core_y = file['mc/core_xy']["Core_y"][i_event] * u.m
                            data.mc.h_first_int = file['mc/h_first_int']["H_first_int"][i_event] * u.m
                        
                # update tels_with_data:
                if tels_with_data_tmp[0] == 1 and tels_with_data_tmp[1] == 0:
                    tels_with_data = {1}
                    time = Time(file['trig/gps_time']["M1_mjd"][i_event] * cds.MJD, file['trig/gps_time']["M1_sec"][i_event] * u.s,
                                  format='unix', scale='utc', precision=9)
                elif tels_with_data_tmp[0] == 0 and tels_with_data_tmp[1] == 1:
                    tels_with_data = {2}
                    time = Time(file['trig/gps_time']["M2_mjd"][i_event] * cds.MJD, file['trig/gps_time']["M2_sec"][i_event] * u.s,
                                  format='unix', scale='utc', precision=9)
                elif tels_with_data_tmp[0] == 1 and tels_with_data_tmp[1] == 1:
                    tels_with_data = {1, 2}
                    time = Time(file['trig/gps_time']["M1_mjd"][i_event] * cds.MJD, (file['trig/gps_time']["M1_sec"][i_event]+file['trig/gps_time']["M2_sec"][i_event] )/2. * u.s,
                                  format='unix', scale='utc', precision=9)
                else:
                    tels_with_data = {}

                data.trig.gps_time = time

                data.r0.tels_with_data = tels_with_data
                data.r1.tels_with_data = tels_with_data
                data.dl0.tels_with_data = tels_with_data
                data.trig.tels_with_trigger = tels_with_data
                data.inst.subarray = magic_subarray

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
