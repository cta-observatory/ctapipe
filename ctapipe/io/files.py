"""
low-level utility functions for dealing with data files
"""

import os
from pathlib import Path
from os.path import basename, splitext, dirname, join
from astropy import log
import numpy as np


def get_file_type(filename):
    """
    Returns a string with the type of the given file (guessed from the
    extension). The '.gz' or '.bz2' compression extensions are
    ignored.

    >>> get_file_type('myfile.fits.gz')
    'fits'

    """
    root, ext = os.path.splitext(filename)
    if ext in ['.gz', '.bz2']:
        ext = os.path.splitext(root)[1]

    ext = ext[1:]  # strip off leading '.'

    # special cases:
    if ext in ['fit', 'FITS', 'FIT']:
        ext = 'fits'

    return ext

# Placed here to avoid error from recursive import
from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source


def targetio_source(filepath, max_events=None):
    """
    Temporary function to return a "source" generator from a targetio file,
    only if targetpipe exists on this python interpreter.

    Parameters
    ----------
    filepath : string
        Filepath for the input targetio file
    max_events : int
        Maximum number of events to read

    Returns
    -------
    source : generator
        A generator that can be iterated over to obtain events, obtained from
        a targetio file.
    """

    # Check targetpipe is installed
    try:
        import importlib
        targetpipe_spec = importlib.util.find_spec("targetpipe")
        found = targetpipe_spec is not None
        if found:
            from targetpipe.io.targetio import targetio_event_source
            return targetio_event_source(filepath, max_events=max_events)
        else:
            raise RuntimeError()
    except RuntimeError:
        log.exception("targetpipe is not installed on this interpreter")
        raise


class InputFile:
    """
    Class to handle generic input files. Enables obtaining the "source"
    generator, regardless of the type of file (either hessio or camera file).

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
    origin : {'hessio', 'targetio'}
        The type of file, related to its source.
        Automatically set from `input_path`.
    output_directory : str
        Directory to save outputs for this file

    """

    def __init__(self, input_path, file_origin):
        """
        Parameters
        ----------
        input_path : str
            Full path to the file
        file_origin : str
            Origin/type of file e.g. hessio, targetio
        """
        self.__input_path = None
        self.directory = None
        self.filename = None
        self.extension = None
        self.origin = file_origin
        self.output_directory = None

        self.input_path = input_path

        log.info("[file] {}".format(self.input_path))
        log.info("[file][origin] {}".format(self.origin))

    @property
    def input_path(self):
        return self.__input_path

    @input_path.setter
    def input_path(self, string):
        path = Path(string)
        try:
            if not path.exists():
                raise FileNotFoundError
        except FileNotFoundError:
            log.exception("file path does not exist: '{}'".format(string))

        self.__input_path = path.as_posix()
        self.directory = dirname(self.__input_path)
        self.filename = splitext(basename(self.__input_path))[0]
        self.extension = splitext(self.__input_path)[1]
        self.output_directory = join(self.directory, self.filename)

    @staticmethod
    def origin_list():
        """
        Returns
        -------
        origins : list
            List of all the origins that have a method for reading
        """
        origins = ['hessio', 'targetio']
        return origins

    def read(self, max_events=None):
        """
        Read the file using the appropriate method depending on the file origin

        Parameters
        ----------
        max_events : int
            Maximum number of events to read

        Returns
        -------
        source : generator
            A generator that can be iterated over to obtain events
        """

        # Obtain relevent source
        log.debug("[file] Reading file...")
        if max_events:
            log.info("[file] Max events being read = {}".format(max_events))
        switch = {
            'hessio':
                lambda: hessio_event_source(get_path(self.input_path),
                                            max_events=max_events),
            'targetio':
                lambda: targetio_source(self.input_path,
                                        max_events=max_events),
        }
        try:
            source = switch[self.origin]()
        except KeyError:
            log.exception("unknown file origin '{}'".format(self.origin))
            raise
        log.debug("[file] Reading complete")

        return source

    def get_event(self, event_req, id_flag=False):
        """
        Loop through events until the requested event is found

        Parameters
        ----------
        event_req : int
            Event index requested
        id_flag : bool
            'event_req' refers to event_id instead of event_index

        Returns
        -------
        event : `ctapipe` event-container

        """
        if not id_flag:
            log.info("[file][read] Finding event index {}...".format(event_req))
        else:
            log.info("[file][read] Finding event id {}...".format(event_req))
        source = self.read()
        for event in source:
            event_id = event.dl0.event_id
            index = event.count if not id_flag else event_id
            if not index == event_req:
                log.debug("[event_id] skipping event: {}".format(event_id))
                continue
            log.info("[file] Event {} found".format(event_req))
            return event
        log.info("[file][read] Event does not exist!")
        return None

    def get_list_of_event_ids(self, max_events=None):
        log.info("[file][read] Building list of event ids...")
        l = []
        source = self.read(max_events)
        if self.origin is 'targetio':
            event = next(source)
            l = range(event.meta.n_events)
        else:
            for event in source:
                l.append(event.dl0.event_id)
        log.info("[file] Number of events = {}".format(len(l)))
        return l

    def find_max_true_npe(self, telescopes=None, max_events=None):
        """
        Loop through events to find the maximum true npe

        Parameters
        ----------
        telescopes : list
            List of telecopes to include. If None, then all telescopes
            are included.
        max_events : int
            Maximum number of events to read

        Returns
        -------
        max_pe : int

        """
        log.info("[file][read] Finding maximum true npe inside file...")
        source = self.read(max_events)
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
                    if np.all(event.mc.tel[tels[0]].photo_electrons == 0):
                        raise KeyError
                except KeyError:
                    log.exception('[chargeres] Source does not contain '
                                  'true charge')
                    raise
            for telid in tels:
                pe = event.mc.tel[telid].photo_electrons
                this_max = np.max(pe)
                if this_max > max_pe:
                    max_pe = this_max
        log.info("[file] Maximum true npe = {}".format(max_pe))

        return max_pe
