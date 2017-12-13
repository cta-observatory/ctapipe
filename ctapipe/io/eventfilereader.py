"""
Handles reading of different event/waveform containing files
"""
from abc import abstractmethod
from os.path import basename, splitext, dirname, join, exists
from traitlets import Unicode, Int, CaselessStrEnum, observe
from copy import deepcopy
from ctapipe.core import Component, Factory
from ctapipe.utils import get_dataset
from ctapipe.io.hessio import hessio_event_source, hessio_get_list_event_ids


class EventFileReader(Component):
    """
    Parent class for EventFileReaders of different sources.

    A new EventFileReader should be created for each type of event file read
    into ctapipe, e.g. simtelarray files are read by the `HessioFileReader`.

    EventFileReader provides a common high-level interface for accessing data
    from different data sources. Creating an EventFileReader for a new data
    source ensures that data can be accessed in a common way, irregardless of
    the data source.

    Attributes
    ----------
    input_path : str
        Path to the input event file.
    max_events : int
        Maximum number of events to loop through in generator
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

        self._num_events = None

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
            self._init_path(new)
        except AttributeError:
            pass

    @observe('max_events')
    def on_max_events_changed(self, change):
        new = change['new']
        try:
            self.log.warning("Change: max_events={}".format(new))
            self._num_events = None
        except AttributeError:
            pass

    @property
    @abstractmethod
    def r1_calibrator(self):
        """
        Abstract property to be defined in child class.

        Name of the `ctapipe.calib.camera.r1.CameraR1Calibrator` to use for
        this `EventFileReader`.

        If the event format has different
        `ctapipe.calib.camera.r1.CameraR1Calibrator` it should use, depending
        on which camera's data is stored in the file, then this should be a
        method to define the correct CameraR1Calibrator to use.

        If the data source is from data level R1 or above,
        return None or 'NullR1Calibrator'.


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

    @abstractmethod
    def read(self, allowed_tels=None):
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

        Returns
        -------
        source : generator
            A generator that can be iterated over to obtain events
        """

    @abstractmethod
    def get_event(self, requested_event, use_event_id=False):
        """
        Abstract method to be defined in child class.

        Obtain a particular event.

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


class HessioFileReader(EventFileReader):

    @property
    def r1_calibrator(self):
        return 'HessioR1Calibrator'

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
        if not self._num_events:
            ids = hessio_get_list_event_ids(self.input_path,
                                            max_events=self.max_events)
            self._num_events = len(ids)
        self.log.info("Number of events inside file = {}"
                      .format(self._num_events))
        return self._num_events

    def read(self, allowed_tels=None):
        """
        Read the file using the appropriate method depending on the file origin

        Parameters
        ----------
        allowed_tels : list[int]
            select only a subset of telescope, if None, all are read. This can
            be used for example emulate the final CTA data format, where there
            would be 1 telescope per file (whereas in current monte-carlo,
            they are all interleaved into one file)

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
                                     requested_event=None,
                                     use_event_id=None)
        self.log.debug("File reading complete")
        return source

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
        source = hessio_event_source(self.input_path,
                                     max_events=self.max_events,
                                     allowed_tels=None,
                                     requested_event=requested_event,
                                     use_event_id=use_event_id)
        event = next(source)
        return deepcopy(event)


# Import the unofficial EventFileReaders so they can be found by the factory
import ctapipe.io.unofficial.eventfilereader


class EventFileReaderFactory(Factory):
    """
    The `EventFileReader` `ctapipe.core.factory.Factory`. This
    `ctapipe.core.factory.Factory` allows the correct
    `EventFileReader` to be obtained for the event file being read.

    This factory tests each EventFileReader by calling
    `EventFileReader.check_file_compatibility` to see which `EventFileReader`
    is compatible with the file.

    Using `EventFileReaderFactory` in a script allows it to be compatible with
    any data source that has an `EventFileReader` defined.

    To use within a `ctapipe.core.tool.Tool`:

    >>> reader = EventFileReaderFactory.produce(config=self.config, tool=self)

    Parameters
    ----------
    config : traitlets.loader.Config
        Configuration specified by config file or cmdline arguments.
        Used to set traitlet values.
        Set to None if no configuration to pass.
    tool : ctapipe.core.Tool or None
        Tool executable that is calling this component.
        Passes the correct logger to the component.
        Set to None if no Tool to pass.
    kwargs

    Attributes
    ----------
    reader : traitlets.CaselessStrEnum
        A string with the `EventFileReader.__name__` of the reader you want to
        use. If left blank, `EventFileReader.check_file_compatibility` will be
        used to find a compatible reader.
    """
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
        return EventFileReaderFactory.__name__

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
