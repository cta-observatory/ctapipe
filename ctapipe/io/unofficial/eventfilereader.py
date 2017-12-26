"""
Location for unofficial-data-format EventFileReaders
"""

from copy import deepcopy
from traitlets import Unicode, observe
from ctapipe.io.eventfilereader import EventFileReader
from ctapipe.utils import get_dataset


class TargetioFileReader(EventFileReader):

    input_path = Unicode(get_dataset("chec_r1.tio"), allow_none=True,
                         help='Path to the input file containing '
                              'events.').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        EventFileReader for the targetio unofficial data format, the data
        format used by cameras containing TARGET modules, such as CHEC for
        the GCT SST. It provides a template for how to create an
        EventFileReader for other unofficial data formats.

        External Software Installation:
        Refer to
        https://forge.in2p3.fr/projects/gct/wiki/Installing_CHEC_Software

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
        try:
            from ctapipe.io.unofficial.targetio.targetio import \
                TargetioExtractor
        except ModuleNotFoundError:
            err = "Cannot find required modules, please follow " \
                  "installation instructions from " \
                  "ctapipe.io.unofficial.eventfilereader.TargetioFileReader"
            self.log.error(err)
            raise

        # TargetioFileReader uses a class which handles the filling of the
        # event containers from the external libraries
        self.extractor_cls = TargetioExtractor
        self.extractor = self.extractor_cls(self.input_path, self.max_events)

    @property
    def r1_calibrator(self):
        return 'TargetioR1Calibrator'

    @observe('input_path')
    def on_input_path_changed(self, change):
        new = change['new']
        try:
            self.log.warning("Change: input_path={}".format(new))
            self._num_events = None
            self._init_path(new)
            self.extractor = self.extractor_cls(new, self.max_events)
        except AttributeError:
            pass

    @staticmethod
    def check_file_compatibility(file_path):
        # Fast method to check if reader can be used for file
        return file_path.endswith('.tio')

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
