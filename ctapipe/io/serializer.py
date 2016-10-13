"""

Serialize ctapipe containers to file
"""

from astropy.table import Table

from ctapipe.core import Component
from ctapipe.core import Container
from ctapipe.core.traits import (Integer, Float, List, Dict, Unicode, TraitError, observe)
from astropy import log
import numpy as np

__all__ = ["Serializer"]


not_writeable_fields = ('tel', 'tels_with_data')


class Serializer: # (Component)
    ''' Context manager to save Containers to file

    For now
    '''

    # outfile = Unicode(help="Output file name").tag(config=True, require=True)
    # writer = Unicode(help="Writer type (fits, img)").tag(config=True, require=True)

    def __init__(self, outfile, format='fits', mode='a', overwrite=False):
        self.outfile = outfile
        self.writer = TableWriter(outfile=outfile, format=format, overwrite=overwrite)
        self._mode = mode

    def __enter__(self):
        self.open_file = open(self.outfile, self._mode)
        log.debug("Opened {0} in mode '{1}'".format(self.outfile, self._mode))
        return self.open_file

    def __exit__(self, *args):
        self.open_file.close()

    def write(self, container):
        self.writer.write(container)

    def write_source(self, source):
        for container in source:
            self.writer.write(container)

    def finish(self):
        self.writer.save()


def is_writeable(key):
    return not(key in not_writeable_fields)

def writeable_items(container):
    d = dict(container.items())
    for k in not_writeable_fields:
        d.pop(k, None)
    return d

def to_table(container):
    names = list()
    columns = list()
    for k, v in writeable_items(container).items():
        names.append(k)
        columns.append(v)

    return names, columns

class TableWriter:
    def __init__(self, outfile, format, overwrite):
        self.table = Table()
        self._created_table = False
        self.format = format
        self.outfile = outfile
        self.overwrite = overwrite

    def _setup_table(self, container):
        '''Create Table from Container'''

        names, columns = to_table(container)
        self.table = Table(rows=[columns], # I need first row to be written here so dtypes can be inferred
                           names=names,
                           meta=container.meta.as_dict())
        # Write HDU name
        self.table.meta["EXTNAME"] = container._name
        self._created_table = True

    def write(self, container):
        if not isinstance(container, Container):
            log.error("Can write only Containers")
            return
        if not self._created_table:
            self._setup_table(container)
        else:
            self.table.add_row(writeable_items(container))

    def save(self, **kwargs):
        '''Write table using astropy.table write method'''
        self.table.write(output=self.outfile, format=self.format, overwrite=self.overwrite, **kwargs)
        return self.table


def main():
    import pickle
    with open('calibrated.pickle', 'rb') as f:
        data = pickle.load(f)

    with Serializer("output.fits") as writer:
        for container in data:
            print(container)
            writer.write(container)

if __name__ == "__main__":
    import pickle

    with open('calibrated.pickle', 'rb') as f:
        data = pickle.load(f)
    container=data[0]
    S = Serializer("output.fits", overwrite=True)

    dl0 = container.dl0
    dl1 = container.dl1
    # print(dl0)
    # print(writeable_items(dl0))
    # names, dtypes, data = to_table(dl0)
    S.write(data[0].dl0)
    S.write(data[1].dl0)
    S.write(data[2].dl0)

    # S.write_source(data)
    print(S.writer.table)

    # with Serializer("output.fits") as writer:
    # for container in data:
    #     print(container)
    #     S.write(container)
    S.finish()
