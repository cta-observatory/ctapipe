"""
Location for unofficial-data-format EventFileReaders
"""

from copy import deepcopy
from traitlets import Unicode, observe
from ctapipe.io.eventfilereader import EventFileReader
from ctapipe.utils import get_dataset, check_modules_installed


targetio_modules = ["target_driver", "target_io", "target_calib"]
if check_modules_installed(targetio_modules):
    from ctapipe.io.unofficial.targetio.targetio import TargetioExtractor

    class TargetioFileReader(EventFileReader):
        name = 'TargetioFileReader'
        origin = 'targetio'

        input_path = Unicode(get_dataset("chec_r1.tio"), allow_none=True,
                             help='Path to the input file containing '
                                  'events.').tag(config=True)

        def __init__(self, config, tool, **kwargs):
            """
            Class to handle targetio input files. Enables obtaining the
            "source" generator.

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
            kwargs
            """
            super().__init__(config=config, tool=tool, **kwargs)

            # TargetioFileReader uses a class which handles the filling of the
            # event containers from the external libraries
            self.extractor = TargetioExtractor(self.input_path,
                                               self.max_events)

        @observe('input_path')
        def on_input_path_changed(self, change):
            new = change['new']
            try:
                self.log.warning("Change: input_path={}".format(new))
                self._num_events = None
                self._init_path(new)
                self.extractor = TargetioExtractor(new, self.max_events)
            except AttributeError:
                pass

        @staticmethod
        def check_file_compatibility(file_path):
            compatible = False
            # Fast method to check if reader can be used for file
            if file_path.endswith('.tio'):
                modules = ["target_driver", "target_io", "target_calib"]
                if check_modules_installed(modules):
                    compatible = True
            return compatible

        @property
        def num_events(self):
            if not self._num_events:
                # Define the fastest method possible to obtain number of events
                # in file. Remember to correct for max_events.
                num = self.extractor.n_events
                if self.max_events and self.max_events < num:
                    num = self.max_events
                self._num_events = num
            return self._num_events

        def read(self, allowed_tels=None):
            self.log.debug("Reading file...")
            if self.max_events:
                self.log.info("Max events being read = {}"
                              .format(self.max_events))
            source = self.extractor.read_generator()
            self.log.debug("File reading complete")
            return source

        def get_event(self, requested_event, use_event_id=False):
            self.extractor.read_event(requested_event, use_event_id)
            event = self.extractor.data
            return deepcopy(event)
