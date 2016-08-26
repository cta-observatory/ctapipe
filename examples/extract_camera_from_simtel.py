"""
Simple example to read some camera geometry info from a
SimTelArray data file (using pyhessio) and write it to a set of FITS tables.
"""
import hessio as h
import sys
from astropy.table import Table
from astropy import units as u
from ctapipe import io

# TODO: use io.fits instead and make the table be variable length
# TODO: make this a tool (ctapipe-eventio2tels)

if __name__ == '__main__':

    filename = sys.argv.pop(1)

    h.file_open(filename)
    event = h.move_to_next_event()
    next(event)

    for telid in range(1, h.get_num_telescope()):
        try:
            px, py = h.get_pixel_position(telid)
            camtab = Table(names=['PIX_POS_X', 'PIX_POS_Y'],
                           data=[px * u.m, py * u.m])
            camtab.meta['N_PIX'] = h.get_num_pixels(telid)
            camtab.meta['N_SAMPS'] = h.get_num_samples(telid)
            camtab.meta['N_TIMES'] = h.get_pixel_timing_num_times_types(telid)
            camtab.meta['MIR_AREA'] = h.get_mirror_area(telid)
            geom = io.guess_camera_geometry(px * u.m, py * u.m)
            camtab.meta['TELCLASS'] = geom.cam_id
            camtab.meta['PIXTYPE'] = geom.pix_type

            filename = "cam_{:03d}.fits".format(telid)
            camtab.write(filename)
            print("WROTE ", filename)
        except h.HessioTelescopeIndexError:
            break
