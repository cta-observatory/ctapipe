"""
Dump instrumental descriptions in a monte-carlo (simtelarray) input file to 
FITS files that can be loaded independently (e.g. with 
CameraGeometry.from_table()).  The name of the output files are 
automatically generated.
"""

from ctapipe.core.traits import (Unicode, Dict, Bool)
from ctapipe.core import Tool
from ctapipe.io.hessio import hessio_event_source
from ctapipe.instrument import CameraGeometry, get_camera_types, print_camera_types


class DumpInstrumentTool(Tool):
    description = Unicode(__doc__)
    name='ctapipe-dump-instrument'

    infile = Unicode(help='input simtelarray file').tag(config=True)
    aliases = Dict(dict(infile='DumpInstrumentTool.infile'))

    def setup(self):
        source = hessio_event_source(self.infile)
        data = next(source)  # get one event, so the instrument table is
                             # filled in
        self.inst = data.inst # keep a pointer to the instrument stuff
        pass

    def start(self):
        self.write_camera_geometries()

    def finish(self):
        pass

    def write_camera_geometries(self):
        cam_types = get_camera_types(self.inst)
        print_camera_types(self.inst, printer=self.log.info)
        for cam_name in cam_types:
            self.log.debug("writing {}".format(cam_name))
            tel_id = cam_types[cam_name].pop()
            pix = self.inst.pixel_pos[tel_id]
            flen = self.inst.optical_foclen[tel_id]
            geom = CameraGeometry.guess(*pix, flen)
            table = geom.to_table()
            table.meta['SOURCE'] = self.infile
            table.write("CTA-{}.camgeom.fits.gz".format(cam_name))


def main():
    tool = DumpInstrumentTool()
    tool.run()

if __name__ == '__main__':
    main()