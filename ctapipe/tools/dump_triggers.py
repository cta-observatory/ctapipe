"""
dump a FITS table ofthe event times and trigger patterns from a simtelarray input file.
"""
import sys
import argparse

import pyhessio
import numpy as np
from astropy.table import Table
from astropy.time import Time
from astropy import units as u

MAX_TELS = 200

def main():

    # get command-line arguments:
    parser=argparse.ArgumentParser()
    parser.add_argument('filename', metavar='SIMTEL_FILE',
                        help='Input simtelarray file')
    parser.add_argument('-o','--output', metavar='FILENAME',
                        help=('output filename (e.g. times.fits), which '
                              'can be any format supported by astropy.table'),
                        default='times.fits.gz')
    args = parser.parse_args()

    # setup output table
    events = Table(names=['EVENT_ID', 'T_REL', 'TRIGGERED_TELS'],
                dtype=[np.int64, np.float64, np.uint8])

    events['TRIGGERED_TELS'].shape = (0, MAX_TELS)
    events['T_REL'].unit = u.s
    events['T_REL'].description = 'Time relative to first event'
    events.meta['INPUT'] = args.filename
    
    trigpattern = np.zeros(MAX_TELS)
    starttime = None
        
    try:
        pyhessio.file_open(args.filename)

        for run_id, event_id in pyhessio.move_to_next_event():

            ts, tns = pyhessio.get_central_event_gps_time()
            gpstime = Time(ts*u.s, tns*u.ns, format='gps', scale='utc')
            if starttime is None:
                starttime = gpstime

            reltime = (gpstime - starttime).sec
                                
            # build the trigger pattern as a fixed-length array
            # (better for storage in FITS format)
            trigtels = pyhessio.get_telescope_with_data_list()
            trigpattern[:] = 0        # zero the trigger pattern
            trigpattern[trigtels] = 1  # set the triggered telescopes to 1

            events.add_row((event_id, reltime, trigpattern))

        events.write(args.output)
        print("Table written to '{}'".format(args.output))
        print(events)

    except Exception as err:
        print("ERROR: {}, stopping".format(err))
        
    finally:
        pyhessio.close_file()
