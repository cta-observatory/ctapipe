from ctapipe.io.eventsource import EventSource
from ctapipe.io.containers import DataContainer
from numpy import stack, zeros, swapaxes, array, int16, uint64
from ctapipe.instrument import TelescopeDescription, SubarrayDescription, \
	OpticsDescription
from ctapipe.instrument.camera import CameraGeometry
from astropy import units as u
from astropy.coordinates import Angle
import numpy as np
import tables

__all__ = ['HiPeHDF5EventSource']
HI_GAIN = 0
LO_GAIN = 1

def _convert_per_events_to_per_telescope(hfile):
	'''
	Concert the telescope storage into a event storage
	Parameters:
	-----------
		hfile : HDF5 file to be used
	Return:
	-------
		dictionary which contains the events with the proper telescopes
	'''
	events=dict()
	for telNode in hfile.walk_nodes('/Tel', 'Group'):
		try:
			tabEventId = telNode.eventId.read()
			tabEventId = tabEventId["eventId"]
			telescopeIndex = uint64(telNode.telIndex.read())
			telescopeId = uint64(telNode.telId.read())
			for i,eventId in enumerate(tabEventId):
				try:
					events[eventId].append((telescopeId, telescopeIndex, i))
				except KeyError:
					events[eventId] = [(telescopeId, telescopeIndex, i)]
		except tables.exceptions.NoSuchNodeError as e:
			#For the Tel dataset only
			pass
	return events


class HiPeHDF5EventSource(EventSource):
	"""
	EventSource for the hiPeHDF5 file format.
	hiPeHDF5 is a data format for Cherenkov Telescope Array (CTA)
	that provides a memory access patterns
	adapted for high performance computing algorithms.
	It allows algorithms to take advantage of the latest SIMD
	(Single input multiple data) operations included in modern processors,
	for native vectorized optimization of analytical
	data processing.
	It provides C++ shared libraries and Python3 library.

	This class utilises `hipedata` library to read the mcrun file, and stores
	the information into the event containers.
	"""

	def __init__(self, config=None, parent=None, **kwargs):
		super().__init__(config=config, parent=parent, **kwargs)

		self.metadata['is_simulation'] = True

		# Create MCRun isntance and load file into memory
		self.run = tables.open_file(self.input_url, "r")
	
	@staticmethod
	def is_compatible(file_path):
		try:
			hfile = tables.open_file(file_path, "r")
			runHeader = hfile.root.RunHeader
			tels = hfile.root.Tel
			corsika = hfile.root.Corsika
			hfile.close()
			return True
		except Exception:
			return False

	def __exit__(self, exc_type, exc_val, exc_tb):
		pass

	def _generator(self):
		# HiPeData arranges data per telescope and not by event like simtel
		# We need to first create a dictionary.
		#   key -> EventId, value -> List of event from triggered telescopes
		self.events = _convert_per_events_to_per_telescope(self.run)

		# the container is initialized once, and data is replaced within
		# it after each yield
		counter = 0
		data = DataContainer()
		data.meta['origin'] = "hipehdf5"

		# some hessio_event_source specific parameters
		data.meta['input_url'] = self.input_url
		data.meta['max_events'] = self.max_events

		'''
		MC data are valid for the whole run
		'''
		data.mc.tel.clear()  # clear the previous telescopes
		for telNode in self.run.walk_nodes('/Tel', 'Group'):
			try:
				tel_id = uint64(telNode.telId.read())
				data.mc.tel[tel_id].dc_to_pe = telNode.tabGain.read()
				data.mc.tel[tel_id].pedestal = telNode.tabPed.read()
				data.mc.tel[tel_id].reference_pulse_shape = telNode.tabRefShape.read()
			except tables.exceptions.NoSuchNodeError as e:
				pass
		
		corsika = self.run.root.Corsika
		tabEvent = corsika.tabCorsikaEvent.read()
		tabEventId = tabEvent["eventId"]
		
		runHeader = self.run.root.RunHeader
		azimuth = runHeader.azimuth.read()
		
		for event_id, event_list in self.events.items():
			if counter == 0:
				# subarray info is only available when an event is loaded,
				# so load it on the first event.
				data.inst.subarray = self._build_subarray_info(self.run)

			obs_id = 0
			tels_with_data = set([info[0] for info in event_list])
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

			#data.trig.tels_with_trigger = set(tels_with_data)
			data.trig.tels_with_trigger = array(list(tels_with_data), dtype=int16)
			
			indexSimu = np.where(tabEventId == 2498406)
			'''
			time_s, time_ns = file.get_central_event_gps_time()
			data.trig.gps_time = Time(time_s * u.s, time_ns * u.ns, format='unix', scale='utc')
			'''
			data.mc.energy = tabEvent["energy"] * u.TeV
			data.mc.alt = Angle(tabEvent["alt"], u.rad)
			data.mc.az = Angle(tabEvent["az"], u.rad)
			data.mc.core_x = tabEvent["coreX"] * u.m
			data.mc.core_y = tabEvent["coreY"] * u.m
			data.mc.h_first_int = tabEvent["h_first_int"] * u.m
			data.mc.x_max = tabEvent["xmax"] * u.g / (u.cm**2)
			data.mc.shower_primary_id = tabEvent["showerPrimaryId"]
			
			data.mcheader.run_array_direction = Angle(azimuth * u.rad)
			
			# this should be done in a nicer way to not re-allocate the
			# data each time (right now it's just deleted and garbage
			# collected)

			data.r0.tel.clear()
			data.r1.tel.clear()
			data.dl0.tel.clear()
			data.dl1.tel.clear()

			for telescopeId, telescopeIndex, event in event_list:
				
				telNode = self.run.get_node("/Tel", 'Tel_' + str(telescopeIndex))
				
				matWaveform = telNode.waveform.read(event, event + 1)
				matWaveform = matWaveform["waveform"]
				
				matSignalPS = matWaveform[0].swapaxes(1, 2)
				data.r0.tel[telescopeId].waveform = matSignalPS
				
				#data.r0.tel[telescopeId].image= matSignalPS.sum(axis=2)
				#data.r0.tel[telescopeId].num_trig_pix = file.get_num_trig_pixels(telescopeId)
				#data.r0.tel[telescopeId].trig_pix_id = file.get_trig_pixels(telescopeId)
				
				
			yield data
			counter += 1
		return

	def _build_subarray_info(self, run):
		"""
		constructs a SubarrayDescription object from the info in an
		MCRun

		Parameters
		----------
		run: MCRun object

		Returns
		-------
		SubarrayDescription :
			instrumental information
		"""
		subarray = SubarrayDescription("MonteCarloArray")
		runHeader = run.root.RunHeader
		
		tabFocalTel = runHeader.tabFocalTel.read()
		tabPosTelX = runHeader.tabPosTelX.read()
		tabPosTelY = runHeader.tabPosTelY.read()
		tabPosTelZ = runHeader.tabPosTelZ.read()
		
		tabPoslXYZ = np.ascontiguousarray(np.vstack((tabPosTelX, tabPosTelY, tabPosTelZ)).T)
		
		'''
		# Correspance HiPeData.Telscope.Type and camera name
		# 0  LSTCam, 1 NectarCam, 2 FlashCam, 3 SCTCam,
		# 4 ASTRICam, 5 DigiCam, 6 CHEC
		'''
		mapping_camera = {0: 'LSTCam', 1: 'NectarCam', 2: 'FlashCam',
						3: 'SCTCam', 4: 'ASTRICam', 5: 'DigiCam',
						6: 'CHEC'}
		
		mapping_telName = {0:'LST', 1:'MST', 2:'MST', 3:'MST', 4:'SST-ASTRI', 5:'SST-1M', 6:'SST-2M'}
		
		for telNode in self.run.walk_nodes('/Tel', 'Group'):
			try:
				telType = uint64(telNode.telType.read())
				telIndex = uint64(telNode.telIndex.read())
				telId = uint64(telNode.telId.read())
				
				cameraName = mapping_camera[telType]
				telName = mapping_telName[telType]
				camera = CameraGeometry.from_name(cameraName)
				camera.cam_id = cameraName
				
				foclen = tabFocalTel[telIndex] * u.m
				
				tel_pos = tabPoslXYZ[telIndex] * u.m
				
				camera.pix_y = telNode.tabPixelX.read() * u.m
				camera.pix_x = telNode.tabPixelY.read() * u.m
				optic = OpticsDescription.from_name(telName)
				optic.equivalent_focal_length = foclen
				telescope_description = TelescopeDescription(telName, telName, optics=optic, camera=camera)

				#tel.optics.mirror_area = mirror_area
				#tel.optics.num_mirror_tiles = num_tiles
				subarray.tels[telId] = telescope_description
				subarray.positions[telId] = tel_pos
			except tables.exceptions.NoSuchNodeError as e:
				pass

		return subarray
