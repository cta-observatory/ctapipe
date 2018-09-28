# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
EventSource for LSTCam protobuf-fits.fz-files.

Needs protozfits v1.4.0 from github.com/cta-sst-1m/protozfitsreader
"""

import numpy as np
from os.path import exists
from ctapipe.core import Provenance
from .eventsource import EventSource
from .containers import LSTDataContainer

__all__ = ['LSTEventSource']


class LSTEventSource(EventSource):

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

        self.multi_file = MultiFiles(self.input_url)
        self.camera_config = self.multi_file.camera_config

        self.log.info("Read {} input files".format(self.multi_file.num_inputs()))


    def _generator(self):

        # container for LST data
        data = LSTDataContainer()
        data.meta['input_url'] = self.input_url

        # fill LST data from the CameraConfig table
        self.fill_lst_service_container_from_zfile(data.lst, self.camera_config)

        # loop on events
        for count, event in enumerate(self.multi_file):

            data.count = count

            # fill specific LST event data
            self.fill_lst_event_container_from_zfile(data.lst, event)

            # fill general R0 data
            self.fill_r0_container_from_zfile(data.r0, event)
            yield data


    @staticmethod
    def is_compatible(file_path):
        from astropy.io import fits
        try:
            # The file contains two tables:
            #  1: CameraConfig
            #  2: Events
            h = fits.open(file_path)[2].header
            ttypes = [
                h[x] for x in h.keys() if 'TTYPE' in x
            ]
        except OSError:
            # not even a fits file
            return False

        except IndexError:
            # A fits file of a different format
            return False

        is_protobuf_zfits_file = (
            (h['XTENSION'] == 'BINTABLE') and
            (h['EXTNAME'] == 'Events') and
            (h['ZTABLE'] is True) and
            (h['ORIGIN'] == 'CTA') and
            (h['PBFHEAD'] == 'R1.CameraEvent')
        )

        is_lst_file = 'lstcam_counters' in ttypes
        return is_protobuf_zfits_file & is_lst_file

    def fill_lst_service_container_from_zfile(self, container, camera_config):

        container.tels_with_data = [camera_config.telescope_id, ]
        svc_container = container.tel[camera_config.telescope_id].svc

        svc_container.telescope_id = camera_config.telescope_id
        svc_container.cs_serial = camera_config.cs_serial
        svc_container.configuration_id = camera_config.configuration_id
        svc_container.date = camera_config.date
        svc_container.num_pixels = camera_config.num_pixels
        svc_container.num_samples = camera_config.num_samples
        svc_container.pixel_ids = camera_config.expected_pixels_id
        svc_container.data_model_version = camera_config.data_model_version

        svc_container.num_modules = camera_config.lstcam.num_modules
        svc_container.module_ids = camera_config.lstcam.expected_modules_id
        svc_container.idaq_version = camera_config.lstcam.idaq_version
        svc_container.cdhs_version = camera_config.lstcam.cdhs_version
        svc_container.algorithms = camera_config.lstcam.algorithms
        svc_container.pre_proc_algorithms = camera_config.lstcam.pre_proc_algorithms




    def fill_lst_event_container_from_zfile(self, container, event):

        event_container = container.tel[self.camera_config.telescope_id].evt

        event_container.configuration_id = event.configuration_id
        event_container.event_id = event.event_id
        event_container.tel_event_id = event.tel_event_id
        event_container.pixel_status = event.pixel_status
        event_container.ped_id = event.ped_id
        event_container.module_status = event.lstcam.module_status
        event_container.extdevices_presence = event.lstcam.extdevices_presence
        event_container.tib_data = event.lstcam.tib_data
        event_container.cdts_data = event.lstcam.cdts_data
        event_container.swat_data = event.lstcam.swat_data
        event_container.counters = event.lstcam.counters
        event_container.chips_flags = event.lstcam.chips_flags
        event_container.first_capacitor_id = event.lstcam.first_capacitor_id
        event_container.drs_tag_status = event.lstcam.drs_tag_status
        event_container.drs_tag = event.lstcam.drs_tag

    def fill_r0_camera_container_from_zfile(self, container, event):

        container.num_samples = self.camera_config.num_samples
        container.trigger_time = event.trigger_time_s
        container.trigger_type = event.trigger_type

        # verify the number of gains
        if event.waveform.shape[0] == (self.camera_config.num_pixels *
                                       container.num_samples):
            n_gains = 1
        elif event.waveform.shape[0] == (self.camera_config.num_pixels *
                                         container.num_samples * 2):
            n_gains = 2
        else:
            raise ValueError("Waveform matrix dimension not supported: "
                             "N_chan x N_pix x N_samples= '{}'"
                             .format(event.waveform.shape[0]))


        container.waveform = np.array(
            (
                event.waveform
            ).reshape(n_gains, self.camera_config.num_pixels, container.num_samples))


    def fill_r0_container_from_zfile(self, container, event):
        container.obs_id = -1
        container.event_id = event.event_id

        container.tels_with_data = [self.camera_config.telescope_id, ]
        r0_camera_container = container.tel[self.camera_config.telescope_id]
        self.fill_r0_camera_container_from_zfile(
            r0_camera_container,
            event
        )


class MultiFiles:
    '''
    In LST they have multiple file writers, which save the incoming events
    into different files, so in case one has 10 events and 4 files,
    it might look like this:
            f1 = [0, 4]
            f2 = [1, 5, 8]
            f3 = [2, 6, 9]
            f4 = [3, 7]
    The task of MultiZFitsFiles is to open these 4 files simultaneously
    and return the events in the correct order, so the user does not really
    have to know about these existence of 4 files.

    In case of multiple input files the name of the files must finish with suffix
    *000.fits.fz, *001.fits.fz, etc... and the user must give as input_url the name
    of the first file (*000.fits.fz). The program will search for the other files.
    In the case of only one input file the input_url can have any form.
    '''

    def __init__(self, input_url):

        self._file = {}
        self._events = {}
        self._events_table = {}
        self._camera_config = {}

        paths = [input_url, ]

        # test how many files are there
        if '000.fits.fz' in input_url:
            i = 0
            while True:
                input_url = input_url.replace(str(i).zfill(3) +
                                              '.fits.fz', str(i + 1)
                                              .zfill(3) + '.fits.fz')
                if exists(input_url):
                    paths.append(input_url)
                    # keep track of all input files
                    Provenance().add_input_file(input_url, role='dl0.sub.evt')
                    i = i + 1
                else:
                    break

        # open the files and get the first fits Tables
        from protozfits import File

        for path in paths:
            self._file[path] = File(path)
            self._events_table[path] = File(path).Events
            try:

                self._events[path] = next(self._file[path].Events)
                self._camera_config[path] = next(self._file[path].CameraConfig)

            except StopIteration:
                pass

        # verify that the configuration_id of all files are the same
        # in the CameraConfig table
        for path in paths:
            assert (self._camera_config[path].configuration_id
                    == self._camera_config[paths[0]].configuration_id)

        # keep the cameraConfig of the first file
        self.camera_config = self._camera_config[paths[0]]

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_event()

    def next_event(self):
        # check for the minimal event id
        if not self._events:
            raise StopIteration

        min_path = min(
            self._events.items(),
            key=lambda item: item[1].event_id,
        )[0]

        # return the minimal event id
        next_event = self._events[min_path]
        try:
            self._events[min_path] = next(self._file[min_path].Events)
        except StopIteration:
            del self._events[min_path]

        return next_event

    def __len__(self):
        total_length = sum(
            len(table)
            for table in self._events_table.values()
        )
        return total_length

    def num_inputs(self):
        return len(self._file)
