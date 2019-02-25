"""
Dump instrumental descriptions in a monte-carlo (simtelarray) input file to 
FITS files that can be loaded independently (e.g. with 
CameraGeometry.from_table()).  The name of the output files are 
automatically generated.
"""

from collections import defaultdict

from ctapipe.core import Tool, Provenance
from ctapipe.core.traits import (Unicode, Dict, Enum)
from ctapipe.io import event_source


def get_camera_types(subarray):
    """ return dict of camera names mapped to a list of tel_ids
     that use that camera

     Parameters
     ----------
     subarray: ctapipe.instrument.SubarrayDescription

     """

    cam_types = defaultdict(list)

    for telid in subarray.tel:
        geom = subarray.tel[telid].camera
        cam_types[geom.cam_id].append(telid)

    return cam_types


class DumpInstrumentTool(Tool):
    description = Unicode(__doc__)
    name = 'ctapipe-dump-instrument'

    infile = Unicode(help='input simtelarray file').tag(config=True)
    format = Enum(['fits', 'ecsv', 'hdf5'],
                  default_value='fits',
                  help='Format of output file',
                  config=True)

    aliases = Dict(dict(infile='DumpInstrumentTool.infile',
                        format='DumpInstrumentTool.format'))

    def setup(self):
        with event_source(self.infile) as source:
            data = next(iter(source))  # get one event, so the instrument table is there

        self.inst = data.inst  # keep a reference to the instrument stuff

    def start(self):
        self.write_camera_geometries()
        self.write_optics_descriptions()
        self.write_subarray_description()

    def finish(self):
        pass

    @staticmethod
    def _get_file_format_info(format_name, table_type, table_name):
        """ returns file extension + dict of required parameters for
        Table.write"""
        if format_name == 'fits':
            return 'fits.gz', dict()
        elif format_name == 'ecsv':
            return 'ecsv.txt', dict(format='ascii.ecsv')
        elif format_name == 'hdf5':
            return 'h5', dict(path="/" + table_type + "/" + table_name)
        else:
            raise NameError("format not supported")

    def write_camera_geometries(self):
        cam_types = get_camera_types(self.inst.subarray)
        self.inst.subarray.info(printer=self.log.info)
        for cam_name in cam_types:
            ext, args = self._get_file_format_info(self.format,
                                                   'CAMGEOM',
                                                   cam_name)

            self.log.debug(f"writing {cam_name}")
            tel_id = cam_types[cam_name].pop()
            geom = self.inst.subarray.tel[tel_id].camera
            table = geom.to_table()
            table.meta['SOURCE'] = self.infile
            filename = f"{cam_name}.camgeom.{ext}"

            try:
                table.write(filename, **args)
                Provenance().add_output_file(filename, 'dl0.tel.svc.camera')
            except IOError as err:
                self.log.warn("couldn't write camera definition '%s' because: "
                              "%s", filename, err)

    def write_optics_descriptions(self):
        sub = self.inst.subarray
        ext, args = self._get_file_format_info(self.format, sub.name, 'optics')

        tab = sub.to_table(kind='optics')
        tab.meta['SOURCE'] = self.infile
        filename = f'{sub.name}.optics.{ext}'
        try:
            tab.write(filename, **args)
            Provenance().add_output_file(filename, 'dl0.sub.svc.optics')
        except IOError as err:
            self.log.warn("couldn't write optics description '%s' because: "
                          "%s", filename, err)

    def write_subarray_description(self):
        sub = self.inst.subarray
        ext, args = self._get_file_format_info(self.format, sub.name,
                                               'subarray')
        tab = sub.to_table(kind='subarray')
        tab.meta['SOURCE'] = self.infile
        filename = f'{sub.name}.subarray.{ext}'
        try:
            tab.write(filename, **args)
            Provenance().add_output_file(filename, 'dl0.sub.svc.subarray')
        except IOError as err:
            self.log.warn("couldn't write subarray description '%s' because: "
                          "%s", filename, err)




def main():
    tool = DumpInstrumentTool()
    tool.run()


if __name__ == '__main__':
    main()
