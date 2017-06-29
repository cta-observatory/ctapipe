"""
Description of Arrays or Subarrays of telescopes
"""

from collections import defaultdict
from astropy.table import Table
from astropy import units as u
import numpy as np

class SubarrayDescription:
    """
    Collects the `TelescopeDescription`s of all telescope along with their
    positions on the ground.

    Parameters
    ----------
    name: str
        name of this subarray
    tel_positions: dict(float)
        dict of telescope positions by tel_id
    tel_descriptions: dict(TelescopeDescription)
        array of TelescopeDescriptions by tel_id
    """

    def __init__(self, name, tel_positions=None, tel_descriptions=None):

        self.name = name
        self.positions = tel_positions or dict()
        self.tels = tel_descriptions or dict()

        assert len(self.positions) == len(self.tels)


    def __str__(self):
        return self.name

    def __repr__(self):
        return "{}(name='{}', num_tels={})".format(
            self.__class__.__name__,
            self.name,
            self.num_tels)

    @property
    def tel(self):
        """ for backward compatibility"""
        return self.tels

    @property
    def num_tels(self):
        return len(self.tels)

    def info(self, printer=print):
        """
        print descriptive info about subarray
        """

        teltypes = defaultdict(list)

        for tel_id, desc in self.tels.items():
            teltypes[str(desc)].append(tel_id)

        printer("Subarray: {}".format(self.name))
        printer("Num Tels: {}".format(self.num_tels))
        printer("")
        printer("                TYPE  Num IDmin  IDmax")
        printer("=====================================")
        for teltype, tels in teltypes.items():
            printer("{:>20s} {:4d} {:4d} ..{:4d}".format(teltype,
                                                         len(tels),
                                                         min(tels),
                                                         max(tels)))
    def to_table(self):
        """ convert to `astropy.table.Table"""
        ids = [x for x in self.tels.keys()]
        descs = [str(x) for x in self.tels.values()]
        pos_x = [x[0].to('m').value for x in self.positions.values()]
        pos_y = [x[1].to('m').value for x in self.positions.values()]

        tab = Table(dict(tel_id=np.array(ids,dtype=np.short),
                         tel_pos_x=np.array(pos_x, dtype=np.float)*u.m,
                         tel_pos_y=np.array(pos_y, dtype=np.float)*u.m,
                         tel_description=descs))
        return tab