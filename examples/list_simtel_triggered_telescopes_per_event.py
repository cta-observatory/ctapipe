#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
List triggered telescopes for each event in a simtel file.
"""

import argparse

import ctapipe
from ctapipe.io.hessio import hessio_event_source


def list_simtel_triggered_telescopes_per_event(simtel_file_path):

    source = hessio_event_source(simtel_file_path, allowed_tels=None, max_events=None)

    tels_per_event_dict = {}   # List of events per telescope

    for event in source:
        tels_per_event_dict[int(event.dl0.event_id)] = [int(tel) for tel in event.trig.tels_with_trigger]

    return tels_per_event_dict


if __name__ == '__main__':

    # PARSE OPTIONS ###########################################################

    desc = "List triggered telescopes for each event in a simtel file"
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

