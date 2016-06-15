#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Print the list of triggered telescopes ID and geometry of the given simtel
file.

Example of output:

    Telescope 001: LSTCam (hexagonal pixels)
    Telescope 002: LSTCam (hexagonal pixels)
    Telescope 003: LSTCam (hexagonal pixels)
    Telescope 004: LSTCam (hexagonal pixels)
    Telescope 005: NectarCam (hexagonal pixels)
    Telescope 006: NectarCam (hexagonal pixels)
    ...
    Telescope 017: FlashCam (hexagonal pixels)
    Telescope 018: FlashCam (hexagonal pixels)
    ...
    Telescope 029: ASTRI (rectangular pixels)
    ...
    Telescope 053: GATE (rectangular pixels)
    ...

"""

import argparse

import ctapipe
from ctapipe.io.hessio import hessio_event_source


def list_telescopes_geometry(simtel_file_path):
    """Print the list of triggered telescopes ID and geometry of the
    'simtel_file_path' file.

    Parameters
    ----------
    simtel_file_path : str
        The path of the simtel file to process.
    """

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


def main():
    """Parse command options (sys.argv)."""

    # PARSE OPTIONS ###########################################################

    desc = "Print the list of triggered telescopes ID and geometry of the given simtel file."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("fileargs", nargs=1, metavar="FILE",
                        help="The simtel file to process")

    args = parser.parse_args()

    simtel_file_path = args.fileargs[0]

    # PRINT THE LIST ##########################################################

    list_telescopes_geometry(simtel_file_path)


if __name__ == '__main__':
    main()
