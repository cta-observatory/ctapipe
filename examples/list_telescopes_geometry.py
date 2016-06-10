#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Display triggered telescopes ID and geometry from a given simtel file
"""

import argparse

import ctapipe
from ctapipe.io.hessio import hessio_event_source


def list_telescopes_geometry(simtel_file_path):

    source = hessio_event_source(simtel_file_path)

    tel_id_set = set()

    for event in source:
        for tel_id in event.dl0.tels_with_data:
            tel_id_set.add(tel_id)

    for tel_id in tel_id_set:
        x, y = event.meta.pixel_pos[tel_id]
        foclen = event.meta.optical_foclen[tel_id]
        geom = ctapipe.io.CameraGeometry.guess(x, y, foclen)
        print("Telescope {:03d}: {} ({} pixels)".format(tel_id, geom.cam_id, geom.pix_type))


if __name__ == '__main__':

    # PARSE OPTIONS ###########################################################

    desc = "Display triggered telescopes ID and geometry from a given simtel file."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("fileargs", nargs=1, metavar="FILE",
                        help="The simtel file to process")

    args = parser.parse_args()

    simtel_file_path = args.fileargs[0]

    # PRINT THE LIST ##########################################################

    list_telescopes_geometry(simtel_file_path)

