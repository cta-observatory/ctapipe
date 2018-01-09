"""
Handles reading of different event/waveform containing files
"""
from abc import abstractmethod
from os.path import exists
from traitlets import Unicode, Int, CaselessStrEnum
from copy import deepcopy
from ctapipe.core import Component, Factory
from ctapipe.utils import get_dataset


class EventFileReader(Component):
    """
    Parent class for EventFileReaders of different sources.

    A new EventFileReader should be created for each type of event file read
    into ctapipe, e.g. sim_telarray files are read by the `HessioFileReader`.

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
        if not exists(self.input_path):
            raise FileNotFoundError("file path does not exist: '{}'"
                                    .format(self.input_path))
        self.log.info("INPUT PATH = {}".format(self.input_path))

        if self.max_events:
            self.log.info("Max events being read = {}".format(self.max_events))

    @staticmethod
    @abstractmethod
    def is_compatible(file_path):
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
    def camera(self):
        """
        Name of the camera contained in the file. Read from the file or
        inferred from the EventFileReader class if the data format is only
        used by one camera.

        This property is used to choose the correct algorithms to process the
        waveforms, as some cameras require different algorithms to others
        (such as with R1 calibration and charge extraction).

        Returns
        -------
        camera_name : str
        """

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        """
        Abstract method to be defined in child class.

        Method where the filling of the `ctapipe.io.containers` occurs.

        Returns
        -------
        data : ctapipe.io.container
            The event container filled with the event information
        """

    def __getitem__(self, item):
        """
        Obtain a particular event

        Parameters
        ----------
        item : int or str
            If `item` is an int, then this is the event_index for the event
            obtained. If `item` is a str, then this is the event_id for the
            event obtained.

        Returns
        -------
        data : ctapipe.io.container
            The event container filled with the requested event's information

        """
        use_event_id = False
        msg = "Event index {} not found in reader".format(item)
        if isinstance(item, str):
            item = int(item)
            use_event_id = True
            msg = "Event id {} not found in reader".format(item)

        if not use_event_id and self.max_events and item >= self.max_events:
            msg = "Event index {} outside of specified max_events {}"\
                .format(item, self.max_events)
        elif not use_event_id:
            for event in self:
                if event.index == item:
                    return deepcopy(event)
        else:
            for event in self:
                if event.id == item:
                    return deepcopy(event)

        raise KeyError(msg)

    def __len__(self):
        if not self._num_events:
            self.log.info("Obtaining number of events in file...")
            count = 0
            for _ in self:
                if count >= self.max_events:
                    break
                count += 1
            self._num_events = count
        return self._num_events


# EventFileReader imports so that EventFileReaderFactory can see them
from ctapipe.io.hessiofilereader import HessioFileReader


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
        return self.__class__.__name__

    def get_product_name(self):
        if self.reader is not None:
            return self.reader
        else:
            if self.input_path is None:
                raise ValueError("Please specify an input_path for event file")
            try:
                for subclass in self.subclasses:
                    if subclass.is_compatible(self.input_path):
                        return subclass.__name__
                raise ValueError
            except ValueError:
                self.log.exception("Cannot find compatible EventFileReader "
                                   "for: {}".format(self.input_path))
                raise
