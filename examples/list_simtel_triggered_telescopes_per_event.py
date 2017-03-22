#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Print the list triggered telescopes per event of the given simtel file.
"""

import argparse
from ctapipe.io.hessio import hessio_event_source


def list_simtel_triggered_telescopes_per_event(simtel_file_path):
    """Make a dictionary of triggered telescopes per event of the
    'simtel_file_path' file.

    Parameters
    ----------
    simtel_file_path : str
        The path of the simtel file to process.

    Returns
    -------
    Dictionary of triggered telescopes per event of the 'simtel_file_path'
    file (for each item: key=event id, value=the list of triggered telescopes).
    """

    source = hessio_event_source(simtel_file_path, allowed_tels=None, max_events=None)

    tels_per_event_dict = {}   # List of events per telescope

    for event in source:
        tels_per_event_dict[int(event.r0.event_id)] = [int(tel) for tel in event.trig.tels_with_trigger]

    return tels_per_event_dict


def main():
    """Parse command options (sys.argv) and print the dictionary of triggered
    telescopes returned by the 'list_simtel_triggered_telescopes_per_event' function."""

    # PARSE OPTIONS ###########################################################

    desc = "Print the list of triggered telescopes per event of the given simtel file."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("fileargs", nargs=1, metavar="FILE",
                        help="The simtel file to process")

    args = parser.parse_args()
    simtel_file_path = args.fileargs[0]

    # PRINT THE LIST ##########################################################

    tels_per_event_dict = list_simtel_triggered_telescopes_per_event(simtel_file_path)

    print("Triggered telescopes per event:")
    for event_id, telescopes_id_list in tels_per_event_dict.items():
        print("- Event {:06}: {}".format(event_id, telescopes_id_list))


if __name__ == '__main__':
    main()
