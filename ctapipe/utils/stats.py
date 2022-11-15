import numpy as np
from scipy.special import erf


def survival_function(x):
    # this 3x faster than scipy.stats.norm.sf
    return 1 - (erf(x / np.sqrt(2.0))) / 2.0
