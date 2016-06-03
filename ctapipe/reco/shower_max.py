from math import log
import numpy as np
from astropy import units as u

from ctapipe.utils.histogram import nDHistogram


class ShowerMaxEstimator:
    def __init__(self, filename):
        altitude  = []
        thickness = []
        atm_file = open(filename, "r")
        for line in atm_file:
            if line.startswith("#"): continue
            altitude .append(float(line.split()[0]))
            thickness.append(float(line.split()[2]))
        
        self.atmosphere = nDHistogram( [np.array(altitude)*u.km], ["altitude"] )
        self.atmosphere.data = (thickness[0:1]+thickness)*u.g * u.cm**-2

        
    def find_shower_max_height(self,energy,h_first_int,gamma_alt):
        # offset of the shower-maximum in radiation lengths
        c = 0.97 * log(energy / (83 * u.MeV)) - 1.32
        # radiation length in dry air at 1 atm = 36,62 g / cm**2 [PDG]
        c *= 36.62 * u.g * u.cm**-2
        # showers with a more horizontal direction spend more path length in each atm. layer
        # the "effective transverse thickness" they have to pass is reduced
        c *= np.sin(gamma_alt)
        
        # find the thickness at the height of the first interaction
        t_first_int = self.atmosphere.interpolate_linear([h_first_int])

        # total thickness at shower maximum = thickness at first interaction + thickness traversed to shower maximum
        t_shower_max = t_first_int + c
        
        # now find the height with the wanted thickness
        for ii, thick1 in enumerate(self.atmosphere.data):
            if t_shower_max > thick1:
                height1 = self.atmosphere.bin_edges[0][ii-1]
                height2 = self.atmosphere.bin_edges[0][ii-2]
                thick2  = self.atmosphere.evaluate([height2])
                
                return (height2-height1) / (thick2-thick1) * (t_shower_max-thick1) + height1
