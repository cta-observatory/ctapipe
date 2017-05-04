"""
Handles reading of different event/waveform containing files
"""
from abc import abstractmethod
from os.path import basename, splitext, dirname, join, exists
import numpy as np
from traitlets import Unicode, Int, CaselessStrEnum, observe
from copy import deepcopy
from ctapipe.core import Component, Factory
from ctapipe.utils import get_dataset
from ctapipe.io.hessio import hessio_event_source, hessio_get_list_event_ids


class EventFileReader(Component):
    """
    Parent class for specific FileReaders

    Attributes
    ----------
    input_path : str
    directory : str
        Automatically set from `input_path`.
    filename : str
        Name of the file without the extension.
        Automatically set from `input_path`.
    extension : str
        Automatically set from `input_path`.
    output_directory : str
        Directory to save outputs for this file

    """
    name = 'EventFileReader'
    origin = None

    input_path = Unicode(get_dataset('gamma_test.simtel.gz'), allow_none=True,
                         help='Path to the input file containing '
                              'events.').tag(config=True)
    max_events = Int(None, allow_none=True,
                     help='Maximum number of events that will be read from'
                          'the file').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        Class to handle generic input files. Enables obtaining the "source"
        generator, regardless of the type of file (either hessio or camera
        file).

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
        super().__init__(config=config, parent=tool, **kwargs)

        if self.origin is None:
            raise ValueError("Subclass of EventFileReader should specify "
                             "an origin")

        self._num_events = None
        self._event_id_list = []

        if self.input_path is None:
            raise ValueError("Please specify an input_path for event file")
        self._init_path(self.input_path)

    def _init_path(self, input_path):
        if not exists(input_path):
            raise FileNotFoundError("file path does not exist: '{}'"
                                    .format(input_path))

        self.input_path = input_path
        self.directory = dirname(input_path)
        self.filename = splitext(basename(input_path))[0]
        self.extension = splitext(input_path)[1]
        self.output_directory = join(self.directory, self.filename)

        if self.log:
            self.log.info("INPUT PATH = {}".format(self.input_path))

    @observe('input_path')
    def on_input_path_changed(self, change):
        new = change['new']
        try:
            self.log.warning("Change: input_path={}".format(new))
            self._num_events = None
            self._event_id_list = []
            self._init_path(new)
        except AttributeError:
            pass

    @observe('origin')
    def on_origin_changed(self, change):
        new = change['new']
        try:
            self.log.warning("Change: origin={}".format(new))
        except AttributeError:
            pass

    @observe('max_events')
    def on_max_events_changed(self, change):
        new = change['new']
        try:
            self.log.warning("Change: max_events={}".format(new))
            self._num_events = None
            self._event_id_list = []
        except AttributeError:
            pass

    @property
    @abstractmethod
    def origin(self):
        """
        Abstract property to be defined in child class.

        Get the name for the origin of the file. E.g. 'hessio'.

        Returns
        -------
        origin : str
        """

    @staticmethod
    @abstractmethod
    def check_file_compatibility(file_path):
        """
        Abstract method to be defined in child class.

        Perform a set of checks to see if the input file is compatible
        with this file reader.

        Parameters
        ----------
        file_path : str
            File path to the event file.

        Returns
        -------
        compatible : bool
            True if file is compatible, False if it is incompatible
        """

    @property
    @abstractmethod
    def num_events(self):
        """
        Abstract property to be defined in child class.

        Obtain the number of events from the file, store it inside
        self._num_events, and return the value.

        Returns
        -------
        self._num_events : int
        """

    @property
    @abstractmethod
    def event_id_list(self):
        """
        Abstract property to be defined in child class.

        Obtain the number of events from the file, store it as
        self._event_id_list, and return the value.

        Returns
        -------
        self._event_id_list : list[self._num_events]
        """

    @abstractmethod
    def read(self, allowed_tels=None, requested_event=None,
             use_event_id=False):
        """
        Abstract method to be defined in child class.

        Read the file using the processes required by the readers file-type.

        Parameters
        ----------
        allowed_tels : list[int]
            select only a subset of telescope, if None, all are read. This can
            be used for example emulate the final CTA data format, where there
            would be 1 telescope per file (whereas in current monte-carlo,
            they are all interleaved into one file)
        requested_event : int
            Seek to a paricular event index
        use_event_id : bool
            If True ,'requested_event' now seeks for a particular event id
            instead of index

        Returns
        -------
        source : generator
            A generator that can be iterated over to obtain events
        """

    def get_event(self, requested_event, use_event_id=False):
        """
        Loop through events until the requested event is found

        Parameters
        ----------
        requested_event : int
            Seek to a paricular event index
        use_event_id : bool
            If True ,'requested_event' now seeks for a particular event id
            instead of index

        Returns
        -------
        event : `ctapipe` event-container

        """
        source = self.read(requested_event=requested_event,
                           use_event_id=use_event_id)
        event = next(source)
        return deepcopy(event)

    def find_max_true_npe(self, telescopes=None):
        """
        Loop through events to find the maximum true npe

        Parameters
        ----------
        telescopes : list
            List of telecopes to include. If None, then all telescopes
            are included.

        Returns
        -------
        max_pe : int

        """
        # TODO: Find an alternate method so this can be removed
        self.log.info("Finding maximum true npe inside file...")
        source = self.read()
        max_pe = 0
        for event in source:
            tels = list(event.dl0.tels_with_data)
            if telescopes is not None:
                tels = []
                for tel in telescopes:
                    if tel in event.dl0.tels_with_data:
                        tels.append(tel)
            if event.count == 0:
                # Check events have true charge included
                try:
                    if np.all(event.mc.tel[tels[0]].photo_electron_image == 0):
                        raise KeyError
                except KeyError:
                    self.log.exception('[chargeres] Source does not contain '
                                       'true charge')
                    raise
            for telid in tels:
                pe = event.mc.tel[telid].photo_electron_image
                this_max = np.max(pe)
                if this_max > max_pe:
                    max_pe = this_max
        self.log.info("Maximum true npe inside file = {}".format(max_pe))

        return max_pe


class HessioFileReader(EventFileReader):
    name = 'HessioFileReader'
    origin = 'hessio'

    @staticmethod
    def check_file_compatibility(file_path):
        compatible = True
        # TODO: Change check to be a try of hessio_event_source?
        if not file_path.endswith('.gz'):
            compatible = False
        return compatible

    @property
    def num_events(self):
        self.log.info("Obtaining number of events in file...")
        if self._num_events:
            pass
        else:
            self._num_events = len(self.event_id_list)
        self.log.info("Number of events inside file = {}"
                      .format(self._num_events))
        return self._num_events

    @property
    def event_id_list(self):
        self.log.info("Retrieving list of event ids...")
        if self._event_id_list:
            pass
        else:
            self.log.info("Building new list of event ids...")
            ids = hessio_get_list_event_ids(self.input_path,
                                            max_events=self.max_events)
            self._event_id_list = ids
        self.log.info("List of event ids retrieved.")
        return self._event_id_list

    def read(self, allowed_tels=None, requested_event=None,
             use_event_id=False):
        """
        Read the file using the appropriate method depending on the file origin

        Parameters
        ----------
        allowed_tels : list[int]
            select only a subset of telescope, if None, all are read. This can
            be used for example emulate the final CTA data format, where there
            would be 1 telescope per file (whereas in current monte-carlo,
            they are all interleaved into one file)
        requested_event : int
            Seek to a paricular event index
        use_event_id : bool
            If True ,'requested_event' now seeks for a particular event id
            instead of index

        Returns
        -------
        source : generator
            A generator that can be iterated over to obtain events
        """

        # Obtain relevent source
        self.log.debug("Reading file...")
        if self.max_events:
            self.log.info("Max events being read = {}".format(self.max_events))
        source = hessio_event_source(self.input_path,
                                     max_events=self.max_events,
                                     allowed_tels=allowed_tels,
                                     requested_event=requested_event,
                                     use_event_id=use_event_id)
        self.log.debug("File reading complete")
        return source


# External Children
try:
    from targetpipe.io.eventfilereader import TargetioFileReader, \
        ToyioFileReader
except ImportError:
    pass


class EventFileReaderFactory(Factory):
    name = "EventFileReaderFactory"
    description = "Obtain EventFileReader based on file type"

    subclasses = Factory.child_subclasses(EventFileReader)
    subclass_names = [c.__name__ for c in subclasses]

    reader = CaselessStrEnum(subclass_names, None, allow_none=True,
                             help='Event file reader to use. If None then '
                                  'a reader will be chosen based on file '
                                  'extension').tag(config=True)

    # Product classes traits
    # Would be nice to have these automatically set...!
    input_path = Unicode(get_dataset('gamma_test.simtel.gz'), allow_none=True,
                         help='Path to the input file containing '
                              'events.').tag(config=True)
    max_events = Int(None, allow_none=True,
                     help='Maximum number of events that will be read from'
                          'the file').tag(config=True)

    def get_factory_name(self):
        return self.name

    def get_product_name(self):
        if self.reader is not None:
            return self.reader
        else:
            if self.input_path is None:
                raise ValueError("Please specify an input_path for event file")
            try:
                for subclass in self.subclasses:
                    if subclass.check_file_compatibility(self.input_path):
                        return subclass.__name__
                raise ValueError
            except ValueError:
                self.log.exception("Cannot find compatible EventFileReader "
                                   "for: {}".format(self.input_path))
                raise
