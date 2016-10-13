"""

Serialize ctapipe containers to file
"""

from astropy.table import Table, Column

from ctapipe.core import Component
from ctapipe.core import Container
from ctapipe.core.traits import (Integer, Float, List, Dict, Unicode, TraitError, observe)
from astropy import log
import numpy as np

__all__ = ["Serializer"]

not_writeable_fields = ('tel', 'tels_with_data', 'calibration_parameters',
'pedestal_subtracted_adc', 'integration_window')


class Serializer:  # (Component)
    ''' Context manager to save Containers to file

    # TODO make it a Component
    '''

    # outfile = Unicode(help="Output file name").tag(config=True, require=True)
    # writer = Unicode(help="Writer type (fits, img)").tag(config=True, require=True)

    def __init__(self, outfile, format='fits', overwrite=False):
        self.outfile = outfile
        self._writer = TableWriter(outfile=outfile, format=format, overwrite=overwrite)

    def __enter__(self):
        log.debug("Serializing on {0}".format(self.outfile))
        return self

    def __exit__(self, *args):
        self..save()

    def write(self, container):
        self._writer.write(container)

    def write_source(self, source):
        raise NotImplementedError
        # for container in source:
        #     self._writer.write(container)

    def save(self):
        self._writer.save()


def is_writeable(key):
    return not (key in not_writeable_fields)


def writeable_items(container):
    # Strip off what we cannot write
    d = dict(container.items())
    for k in not_writeable_fields:
        log.debug("Cannot write column {0}".format(k))
        d.pop(k, None)
    return d


def to_table(container):

    names = list()
    columns = list()
    for k, v in writeable_items(container).items():
        v_arr = np.array(v)
        v_arr = v_arr.reshape((1,) + v_arr.shape)
        log.debug("Creating column for item '{0}' of shape {1}".format(k, v_arr.shape))
        names.append(k)
        columns.append(Column(v_arr))

    return names, columns


class TableWriter:
    def __init__(self, outfile, format, overwrite):
        self.table = Table()
        self._created_table = False
        self.format = format
        self.outfile = outfile
        self.overwrite = overwrite

    def _setup_table(self, container):
        # Create Table from Container

        names, columns = to_table(container)
        self.table = Table(data=columns,  # dtypes are inferred by columns
                           names=names,
                           meta=container.meta.as_dict())
        # Write HDU name
        if self.format == "fits":
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
        # Write table using astropy.table write method
        self.table.write(output=self.outfile, format=self.format, overwrite=self.overwrite, **kwargs)
        return self.table
