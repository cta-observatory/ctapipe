"""
Classes and functions related to telescope Optics
"""

import logging
from ..utils import get_table_dataset
import numpy as np
import astropy.units as u

logger = logging.getLogger(__name__)


class OpticsDescription:
    """
    Describes the optics of a Cherenkov Telescope mirror

    The string representation of an `OpticsDescription` will be a combination
    of the telescope-type and sub-type as follows: "type-subtype". You can
    also get each individually.

    The `OpticsDescription.guess()` constructor can be used to fill in info
    from metadata, e.g. for Monte-Carlo files.

    Parameters
    ----------
    mirror_type: str
        'SC' or 'DC'
    tel_type: str
        'SST', 'MST','LST'
    tel_subtype: str
        subtype of telescope, e.g. '1M' or 'ASTRI'
    equivalent_focal_length: Quantity(float)
        effective focal-length of telescope, independent of which type of
        optics (as in the Monte-Carlo)
    mirror_area: float
        total reflective surface area of the optical system (in m^2)
    num_mirror_tiles: int
        number of mirror facets

    Raises
    ------
    ValueError:
        if tel_type or mirror_type are not one of the accepted values
    TypeError, astropy.units.UnitsError:
        if the units of one of the inputs are missing or incompatible
    """

    def __init__(self, mirror_type, tel_type,
                 tel_subtype, equivalent_focal_length,
                 mirror_area=None, num_mirror_tiles=None):

        if tel_type not in ['LST', 'MST', 'SST']:
            raise ValueError("Unknown tel_type %s", tel_type)

        if mirror_type not in ['SC', 'DC']:
            raise ValueError("Unknown mirror_type: %s", mirror_type)

        self.mirror_type = mirror_type
        self.tel_type = tel_type
        self.tel_subtype = tel_subtype
        self.equivalent_focal_length = equivalent_focal_length.to(u.m)
        self.mirror_area = mirror_area

        self.num_mirror_tiles = num_mirror_tiles

    @classmethod
    def from_name(cls, name, optics_table='optics'):
        """
        Construct an OpticsDescription from the name. This is loaded from
        `optics.fits.gz`, which should be in `ctapipe_resources` or in a
        directory listed in `CTAPIPE_SVC_PATH`.

        Parameters
        ----------
        name: str
            string representation of optics (MST, LST, SST-1M, SST-ASTRI,...)
        optics_table: str
            base filename of optics table if not 'optics.*'


        Returns
        -------
        OpticsDescription

        """
        table = get_table_dataset(optics_table, role='dl0.tel.svc.optics')
        mask = table['tel_description'] == name

        if 'equivalent_focal_length' in table.colnames:
            flen = table['equivalent_focal_length'][mask].quantity[0]
        else:
            flen = table['effective_focal_length'][mask].quantity[0]
            logger.warning("Optics table format out of date: "
                           "'effective_focal_length' "
                           "should be 'equivalent_focal_length'")
        subtype = str(table['tel_subtype'][mask][0])
        if subtype == '--':
            subtype = ''

        optics = cls(
            mirror_type=table['mirror_type'][mask][0],
            tel_type=str(table['tel_type'][mask][0]),
            tel_subtype=subtype,
            equivalent_focal_length=flen,
            mirror_area=table['mirror_area'][mask].quantity[0],
            num_mirror_tiles=table['num_mirror_tiles'][mask][0],
        )
        return optics

    @classmethod
    def get_known_optics_names(cls, optics_table='optics'):
        table = get_table_dataset(optics_table, 'get_known_optics')
        return np.array(table['tel_description'])

    @property
    def identifier(self):
        """ returns a tuple of (tel_type, tel_subtype).  Use str(optics) to
        get a text-based identifier."""
        return self.tel_type, self.tel_subtype

    def info(self, printer=print):
        printer(f'OpticsDescription: "{self}"')
        printer(f'    - mirror_type: {self.mirror_type}')
        printer(f'    - num_mirror_tiles: {self.num_mirror_tiles}')
        printer(f'    - mirror_area: {self.mirror_area}')

    def __repr__(self):
        return "{}(tel_type='{}', tel_subtype='{}')".format(
            str(self.__class__.__name__), self.tel_type, self.tel_subtype)

    def __str__(self):
        if self.tel_subtype != '':
            return f"{self.tel_type}-{self.tel_subtype}"
        else:
            return self.tel_type
