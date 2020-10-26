from enum import Enum, auto


class DataLevel(Enum):
    """Enum of the different Data Levels"""

    R0 = auto()
    R1 = auto()
    R2 = auto()
    DL0 = auto()
    DL1_IMAGES = auto()
    DL1_PARAMETERS = auto()
    DL2 = auto()
    DL3 = auto()
