from math import log
import numpy as np
from astropy import units as u
from scipy import ndimage

from ctapipe.utils.fitshistogram import Histogram 

class ShowerMaxEstimator:
    def __init__(self, filename, col_altitude=0, col_thickness=2):
        """ small class that calculates the height of the shower maximum
            given a parametrisation of the atmosphere 
            and certain parameters of the shower itself
            
            Parameters:
            -----------
            filename : string
                path to text file that contains a table of the atmosphere parameters
            col_altitude : int
                column in the text file that contains the altitude/height
            col_thickness : int
                column in the text file that contains the thickness
        """
        
        altitude  = []
        thickness = []
        atm_file = open(filename, "r")
        for line in atm_file:
            if line.startswith("#"): continue
            altitude .append(float(line.split()[col_altitude]))
            thickness.append(float(line.split()[col_thickness]))
        
        self.atmosphere = Histogram(axisNames=["altitude"])
        self.atmosphere.hist = thickness*u.g * u.cm**-2
        self.atmosphere.bin_lower_edges = [np.array(altitude)*u.km]

    def interpolate(self, arg, outlierValue=0.,order=3):
        
        axis = self.atmosphere._binLowerEdges[0]
        bin_u = np.digitize(arg.to(axis.unit), axis)
        bin_l = bin_u - 1
        
        unit = arg.unit
        argv = arg.value

        bin_u_edge = axis[ bin_u ].to(unit).value
        bin_l_edge = axis[ bin_l ].to(unit).value
        coordinate =  (argv-bin_u_edge) / (bin_u_edge-bin_l_edge) * (bin_u - bin_l) + bin_u

        return ndimage.map_coordinates(self.atmosphere.hist, [[coordinate]],order=order,cval=outlierValue)[0] * self.atmosphere.hist.unit

        
    def find_shower_max_height(self,energy,h_first_int,gamma_alt):
        """ estimates the height of the shower maximum in the atmosphere
            according to equation (3) in [arXiv:0907.2610v3]
        
        Parameters:
        -----------
        energy : astropy.Quantity
            energy of the parent gamma photon
        h_first_int : astropy.Quantity
            hight of the first interaction
        gamma_alt : astropy.Quantity or float
            altitude / pi-minus-zenith (in radians in case of float) of the parent gamma photon
        
        Returns:
        --------
        shower_max_height : astropy.Quantity
            height of the shower maximum
        """
        
        # offset of the shower-maximum in radiation lengths
        c = 0.97 * log(energy / (83 * u.MeV)) - 1.32
        # radiation length in dry air at 1 atm = 36,62 g / cm**2 [PDG]
        c *= 36.62 * u.g * u.cm**-2
        # showers with a more horizontal direction spend more path length in each atm. layer
        # the "effective transverse thickness" they have to pass is reduced
        c *= np.sin(gamma_alt)
        
        # find the thickness at the height of the first interaction
        t_first_int = self.interpolate(h_first_int)
        
        # total thickness at shower maximum = thickness at first interaction + thickness traversed to shower maximum
        t_shower_max = t_first_int + c
        
        # now find the height with the wanted thickness
        for ii, thick1 in enumerate(self.atmosphere.hist):
            if t_shower_max > thick1:
                height1 = self.atmosphere.bin_lower_edges[0][ii]
                height2 = self.atmosphere.bin_lower_edges[0][ii-1]
                thick2  = self.atmosphere.get_value([height2.to(self.atmosphere._binLowerEdges[0].unit).value])[0]
                
                return (height2-height1) / (thick2-thick1) * (t_shower_max-thick1) + height1
