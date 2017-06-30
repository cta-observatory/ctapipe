"""
Classes and functions related to telescope Optics
"""

import logging

logger = logging.getLogger(__name__)

_FOCLEN_TO_TEL_INFO = {
    # foclen: tel_type, tel_subtype, mirror_type
    28.0: ('LST', '', 'DC'),
    16.0: ('MST', '', 'DC'),
    2.28: ('SST', 'GCT', 'SC'),
    2.15: ('SST', 'ASTRI', 'SC'),
    5.6: ('SST', '1M', 'SC'),
    5.58: ('MST', 'SCT', 'SC'),
    5.59: ('MST', 'SCT', 'SC')

}


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
    effective_focal_length: Quantity(float)
        effective focal-length of telescope, independent of which type of
        optics (as in the Monte-Carlo)
    """

    def __init__(self, mirror_type, tel_type,
                 tel_subtype, effective_focal_length,
                 mirror_area=None, num_mirror_tiles=None):

        if tel_type not in ['LST', 'MST', 'SST']:
            raise ValueError("Unknown tel_type %s", tel_type)

        self.mirror_type = mirror_type
        self.tel_type = tel_type
        self.tel_subtype = tel_subtype
        self.effective_focal_length = effective_focal_length
        self.mirror_area = mirror_area
        self.num_mirror_tiles = num_mirror_tiles

    @classmethod
    def guess(cls, effective_focal_length):
        """
        Construct an OpticsDescription by guessing from metadata (e.g. when
        using a simulation where the exact type is not known)

        Parameters
        ----------
        effective_focal_length: Quantity('m')
            effective optical focal-length in meters

        """

        tel_type, tel_subtype, mir_type = \
            telescope_info_from_metadata(effective_focal_length)

        if tel_type == "unknown":
            logger.warning(("No OpticsDescription found for focal-length "
                            "%s, setting to 'unknown'"),
                           effective_focal_length)

        return cls(mirror_type=mir_type,
                   tel_type=tel_type,
                   tel_subtype=tel_subtype,
                   effective_focal_length=effective_focal_length)

    @property
    def identifier(self):
        """ returns a tuple of (tel_type, tel_subtype).  Use str(optics) to
        get a text-based identifier."""
        return (self.tel_type, self.tel_subtype)

    def __repr__(self):
        return "{}(tel_type='{}', tel_subtype='{}')".format(
            str(self.__class__.__name__), self.tel_type, self.tel_subtype)

    def __str__(self):
        if self.tel_subtype != '':
            return "{}-{}".format(self.tel_type, self.tel_subtype)
        else:
            return self.tel_type


def telescope_info_from_metadata(focal_length):
    """
    helper func to return telescope and mirror info based on metadata

    Parameters
    ----------
    focal_length: float
        effective focal length

    Returns
    -------
    str,str,str:
        tel_type ('LST', 'MST' or 'SST'),
        tel_subtype (model),
        mirror_type ('SC' or 'DC')

    Raises:
    -------
    KeyError:
       if unable to find optics type
    """
    return _FOCLEN_TO_TEL_INFO[round(focal_length.to('m').value, 2)]
