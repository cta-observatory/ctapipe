"""
Dump instrumental descriptions in a monte-carlo (simtelarray) input file to 
FITS files that can be loaded independently (e.g. with 
CameraGeometry.from_table()).  The name of the output files are 
automatically generated.
"""

from ctapipe.core.traits import (Unicode, Dict, Bool, Enum)
from ctapipe.core import Tool
from ctapipe.io.hessio import hessio_event_source
from ctapipe.instrument import CameraGeometry, get_camera_types, print_camera_types


class DumpInstrumentTool(Tool):
    description = Unicode(__doc__)
    name='ctapipe-dump-instrument'

    infile = Unicode(help='input simtelarray file').tag(config=True)
    format = Enum(['fits', 'ecsv', 'hdf5'], default_value='fits', help='Format '
                                                                       'of '
                                                                       'output '
                                                                       'file',
                  config=True)

    aliases = Dict(dict(infile='DumpInstrumentTool.infile',
                        format='DumpInstrumentTool.format'))


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

    def _get_file_format_info(self, format_name, table_type, cam_name):
        """ returns file extension + dict of required parameters for 
        Table.write"""
        if format_name == 'fits':
            return 'fits.gz', dict()
        elif format_name == 'ecsv':
            return 'ecsv.txt', dict(format='ascii.ecsv')
        elif format_name == 'hdf5':
            return 'h5', dict(path="/"+table_type+"/"+cam_name)
        else:
            raise NameError("format not supported")

    def write_camera_geometries(self):
        cam_types = get_camera_types(self.inst)
        print_camera_types(self.inst, printer=self.log.info)
        for cam_name in cam_types:
            ext, args = self._get_file_format_info(self.format,
                                                   'CAMGEOM',
                                                   cam_name)
            self.log.debug("writing {}".format(cam_name))
            tel_id = cam_types[cam_name].pop()
            pix = self.inst.pixel_pos[tel_id]
            flen = self.inst.optical_foclen[tel_id]
            geom = CameraGeometry.guess(*pix, flen)
            table = geom.to_table()
            table.meta['SOURCE'] = self.infile
            table.write("{}.camgeom.{}".format(cam_name, ext), **args)


def main():
    tool = DumpInstrumentTool()
    tool.run()

if __name__ == '__main__':
    main()