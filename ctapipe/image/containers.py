from numpy import nan
from ..core import Container, Field
from .hillas import HillasParametersContainer
from .leakage import LeakageContainer
from .timing_parameters import TimingParametersContainer
from .concentration import ConcentrationContainer


class MorphologyContainer(Container):
    """ Parameters related to pixels surviving image cleaning """

    num_pixels = Field(nan, "Number of usable pixels")
    num_islands = Field(nan, "Number of distinct islands in the image")
    num_small_islands = Field(nan, "Number of <= 2 pixel islands")
    num_medium_islands = Field(nan, "Number of 2-50 pixel islands")
    num_large_islands = Field(nan, "Number of > 10 pixel islands")


class ImageParametersContainer(Container):
    """ Collection of image parameters """

    container_prefix = "params"
    hillas = Field(HillasParametersContainer(), "Hillas Parameters")
    timing = Field(TimingParametersContainer(), "Timing Parameters")
    leakage = Field(LeakageContainer(), "Leakage Parameters")
    concentration = Field(ConcentrationContainer(), "Concentration Parameters")
    morphology = Field(MorphologyContainer(), "Morphology Parameters")
