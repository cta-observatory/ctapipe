from functools import lru_cache

from . import HDF5TableWriter
from ..core import Component
from ..core import traits as tr


class DL1Writer(Component):
    outfile = tr.Unicode('dl1.h5', help='output HDF5 file').tag(config=True)
    groupname = tr.Unicode('DL1', help='HDF5 group name')

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # alternativelly, tables.Filters(complevel=5, complib='blosc')
        filters = None

        self.writer = HDF5TableWriter(self.outfile,
                                      group_name=self.groupname,
                                      num_rows_expected=1000,
                                      filters=filters)

        # exclude the extracted_samples and cleaned columns from all tables
        for ii in range(100):
            self.writer.exclude(self.get_table_name(ii), 'extracted_samples')
            self.writer.exclude(self.get_table_name(ii), 'cleaned')

    @lru_cache(512)
    def get_table_name(self, tel_id):
        return "tel_{:03d}".format(tel_id)

    def write(self, event):

        self.writer.write(table_name='MC', containers=event.mc)

        for tel_id, cont in event.dl1.tel.items():
            table_name = self.get_table_name(tel_id)
            self.writer.write(table_name=table_name, containers=cont)
