#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
List registered events for each telescope in a given simtel file.
"""

import argparse

import ctapipe
from ctapipe.io.hessio import hessio_event_source


def list_simtel_events_per_telescope(simtel_file_path):

    source = hessio_event_source(simtel_file_path, allowed_tels=None, max_events=None)

    events_per_tel_dict = {}   # List of events per telescope

    for event in source:
        triggered_telescopes_list = [int(tel_id) for tel_id in event.trig.tels_with_trigger]
        for telescope_id in triggered_telescopes_list:
            if telescope_id not in events_per_tel_dict:
                events_per_tel_dict[telescope_id] = []
            events_per_tel_dict[telescope_id].append(int(event.dl0.event_id))

    return events_per_tel_dict


if __name__ == '__main__':

    # PARSE OPTIONS ###########################################################

    desc = "List registered events for each telescope in a given simtel file"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("fileargs", nargs=1, metavar="FILE",
                        help="The simtel file to process")

    args = parser.parse_args()
    simtel_file_path = args.fileargs[0]

    # PRINT THE LIST ##########################################################

    events_per_tel_dict = list_simtel_events_per_telescope(simtel_file_path)

    print("Events per telescope:")
    for telescope_id, events_id_list in events_per_tel_dict.items():
        print("- Telescope {:03}: {}".format(telescope_id, events_id_list))

