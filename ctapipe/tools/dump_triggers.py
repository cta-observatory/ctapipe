"""
dump a table of the event times and trigger patterns from a
simtelarray input file.
"""

import numpy as np
from astropy import units as u
from astropy.table import Table

from ctapipe.io import event_source
from ctapipe.core import Provenance, ToolConfigurationError
from ctapipe.core.traits import (Unicode, Dict, Bool)
from ..core import Tool

MAX_TELS = 1000


class DumpTriggersTool(Tool):
    description = Unicode(__doc__)
    name = 'ctapipe-dump-triggers'

    # =============================================
    # configuration parameters:
    # =============================================

    infile = Unicode(help='input simtelarray file').tag(config=True)

    outfile = Unicode('triggers.fits',
                      help='output filename (*.fits, *.h5)').tag(config=True)

    overwrite = Bool(False,
                     help="overwrite existing output file"
                     ).tag(config=True)

    # =============================================
    # map low-level options to high-level command-line options
    # =============================================

    aliases = Dict({'infile': 'DumpTriggersTool.infile',
                    'outfile': 'DumpTriggersTool.outfile'})

    flags = Dict({'overwrite': ({'DumpTriggersTool': {'overwrite': True}},
                                'Enable overwriting of output file')})

    examples = ('ctapipe-dump-triggers --infile gamma.simtel.gz '
                '--outfile trig.fits --overwrite'
                '\n\n'
                'If you want to see more output, use --log_level=DEBUG')

    # =============================================
    # The methods of the Tool (initialize, start, finish):
    # =============================================

    def add_event_to_table(self, event):
        """
        add the current hessio event to a row in the `self.events` table
        """
        gpstime = event.trig.gps_time

        if self._prev_gpstime is None:
            self._prev_gpstime = gpstime

        if self._current_starttime is None:
            self._current_starttime = gpstime

        relative_time = gpstime - self._current_starttime
        delta_t = gpstime - self._prev_gpstime
        self._prev_gpstime = gpstime

        # build the trigger pattern as a fixed-length array
        # (better for storage in FITS format)
        # trigtels = event.get_telescope_with_data_list()
        trigtels = event.dl0.tels_with_data
        self._current_trigpattern[:] = 0  # zero the trigger pattern
        self._current_trigpattern[list(trigtels)] = 1  # set the triggered tels
        # to 1

        # insert the row into the table
        self.events.add_row((event.dl0.event_id, relative_time.sec, delta_t.sec,
                             len(trigtels),
                             self._current_trigpattern))

    def setup(self):
        """ setup function, called before `start()` """

        if self.infile == '':
            raise ToolConfigurationError("No 'infile' parameter was specified. ")

        self.events = Table(names=['EVENT_ID', 'T_REL', 'DELTA_T',
                                   'N_TRIG', 'TRIGGERED_TELS'],
                            dtype=[np.int64, np.float64, np.float64,
                                   np.int32, np.uint8])

        self.events['TRIGGERED_TELS'].shape = (0, MAX_TELS)
        self.events['T_REL'].unit = u.s
        self.events['T_REL'].description = 'Time relative to first event'
        self.events['DELTA_T'].unit = u.s
        self.events.meta['INPUT'] = self.infile

        self._current_trigpattern = np.zeros(MAX_TELS)
        self._current_starttime = None
        self._prev_gpstime = None

    def start(self):
        """ main event loop """
        with event_source(self.infile) as source:
            for event in source:
                self.add_event_to_table(event)

    def finish(self):
        """
        finish up and write out results (called automatically after
        `start()`)
        """
        # write out the final table
        try:
            if self.outfile.endswith('fits') or self.outfile.endswith('fits.gz'):
                self.events.write(self.outfile, overwrite=self.overwrite)
            elif self.outfile.endswith('h5'):
                self.events.write(self.outfile, path='/events',
                                  overwrite=self.overwrite)
            else:
                self.events.write(self.outfile)

            Provenance().add_output_file(self.outfile)
        except IOError as err:
            self.log.warn("Couldn't write output (%s)", err)

        self.log.info('\n %s', self.events)


def main():
    tool = DumpTriggersTool()
    tool.run()
