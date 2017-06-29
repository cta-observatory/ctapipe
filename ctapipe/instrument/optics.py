"""
Classes and functions related to telescope Optics
"""

import logging

logger = logging.getLogger(__name__)

_FOCLEN_TO_TEL_INFO = {
    # foclen: tel_type, tel_subtype, mirror_type
    28.0: ('LST', '', 'DC'),
    16.0: ('MST', '', 'DC'),
    2.28: ('SST', '1m', 'DC'),
    2.15: ('SST', 'GATE', 'SC'),
    5.58: ('MST', 'SCT', 'SC')
}


class OpticsDescription:
    """
    Describes the optics of a Cherenkov Telescope mirror
    """

    def __init__(self, mirror_type, tel_type,
                 tel_subtype, effective_focal_length):
        self.mirror_type = mirror_type
        self.tel_type = tel_type
        self.tel_subtype = tel_subtype
        self.effective_focal_length = effective_focal_length


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
                            "%s, setting to 'unknown'") ,
                           effective_focal_length)

        return cls(mirror_type=mir_type,
                   tel_type=tel_type,
                   tel_subtype=tel_subtype,
                   effective_focal_length=effective_focal_length)





def telescope_info_from_metadata(focal_length):
    """

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
    """
    global _FOCLEN_TO_TEL_INFO
    return _FOCLEN_TO_TEL_INFO.get(round(focal_length.to('m').value, 2),
                                   ('unknown', 'unknown', 'unknown'))
