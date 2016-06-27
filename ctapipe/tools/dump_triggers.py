"""
dump a FITS table of the event times and trigger patterns from a
simtelarray input file.
"""
import sys
import argparse

import pyhessio
import numpy as np
from astropy.table import Table
from astropy.time import Time
from astropy import units as u
from ctapipe.core import Tool
from traitlets import Unicode, Dict, Bool

MAX_TELS = 1000


class DumpTriggersTool(Tool):
    description = __doc__

    # configuration parameters:
    infile = Unicode('', help='input simtelarray file').tag(config=True)
    outfile = Unicode('triggers.fits',
                     help=('output filename, '
                           'Can be any file type supported'
                           'by astropy.table')).tag(config=True)
    overwrite = Bool(False,
                     help="overwrite existing output file").tag(config=True)

    # set which ones are high-level command-line options
    aliases = Dict(dict(log_level='DumpTriggersTool.log_level',
                        infile='DumpTriggersTool.infile',
                        outfile='DumpTriggersTool.outfile'))

    flags = dict(overwrite=({'DumpTriggersTool': {'overwrite': True}},
                            'Enable overwriting of output file'))

    examples = ('ctapipe-dump-triggers --infile gamma.simtel.gz '
                '--outfile trig.fits --overwrite')

    def start(self):

        events = Table(names=['EVENT_ID', 'T_REL', 'N_TRIG', 'TRIGGERED_TELS'],
                       dtype=[np.int64, np.float64, np.int32, np.uint8])

        events['TRIGGERED_TELS'].shape = (0, MAX_TELS)
        events['T_REL'].unit = u.s
        events['T_REL'].description = 'Time relative to first event'
        events.meta['INPUT'] = self.infile

        trigpattern = np.zeros(MAX_TELS)
        starttime = None

        try:
            pyhessio.file_open(self.infile)

            for run_id, event_id in pyhessio.move_to_next_event():

                ts, tns = pyhessio.get_central_event_gps_time()
                gpstime = Time(ts * u.s, tns * u.ns, format='gps', scale='utc')
                if starttime is None:
                    starttime = gpstime

                reltime = (gpstime - starttime).sec

                # build the trigger pattern as a fixed-length array
                # (better for storage in FITS format)
                trigtels = pyhessio.get_telescope_with_data_list()
                trigpattern[:] = 0        # zero the trigger pattern
                trigpattern[trigtels] = 1  # set the triggered telescopes to 1

                events.add_row((event_id, reltime, len(trigtels), trigpattern))

            events.write(self.outfile, overwrite=self.overwrite)
            self.log.info("Table written to '{}'".format(self.outfile))
            self.log.info(events)

        except Exception as err:
            self.log.error("ERROR: {}, stopping".format(err))

        finally:
            pyhessio.close_file()


def main():

    tool = DumpTriggersTool()
    tool.run()
