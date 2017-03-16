#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Print the list of triggered events per telescope of the given simtel file.
"""

import argparse
from ctapipe.io.hessio import hessio_event_source


def list_simtel_events_per_telescope(simtel_file_path):
    """Make a dictionary of triggered events per telescope of the
    'simtel_file_path' file.

    Parameters
    ----------
    simtel_file_path : str
        The path of the simtel file to process.

    Returns
    -------
    Dictionary of triggered events per telescope of the 'simtel_file_path'
    file (for each item: key=telescope id, value=the list of triggered events).
    """

    source = hessio_event_source(simtel_file_path, allowed_tels=None, max_events=None)

    events_per_tel_dict = {}   # List of events per telescope

    for event in source:
        triggered_telescopes_list = [int(tel_id) for tel_id in event.trig.tels_with_trigger]
        for telescope_id in triggered_telescopes_list:
            if telescope_id not in events_per_tel_dict:
                events_per_tel_dict[telescope_id] = []
            events_per_tel_dict[telescope_id].append(int(event.r0.event_id))

    return events_per_tel_dict


def main():
    """Parse command options (sys.argv) and print the dictionary of triggered
    events returned by the 'list_simtel_events_per_telescope' function."""

    # PARSE OPTIONS ###########################################################

    desc = "Print the list of triggered events per telescope of the given simtel file."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("fileargs", nargs=1, metavar="FILE",
                        help="The simtel file to process")

    args = parser.parse_args()
    simtel_file_path = args.fileargs[0]

    # PRINT THE LIST ##########################################################

    events_per_tel_dict = list_simtel_events_per_telescope(simtel_file_path)

    print("Triggered events per telescope:")
    for telescope_id, events_id_list in events_per_tel_dict.items():
        print("- Telescope {:03}: {}".format(telescope_id, events_id_list))


if __name__ == '__main__':
    main()
