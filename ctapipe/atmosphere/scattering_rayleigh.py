#!/bin/env python

## @file
#  The Rayleigh scattering class
# 

import math

####################################
## @brief Class to manage Rayleigh scattering
#
class pRayleigh(object):

    # Constants
    HR0=7.996            # rayleigh scale height (km) ?
    HTHEMIS=1.65         # altitude of the Themis site 1650 m (same as HESS site)
    C = 1.08574          # magnitude vs optical depth

    ####################################
    ## @brief Constructor
    #
    ## @param self
    #  the object instance
    ## @param costheta
    #  the cosine of the theta angle
    ## @param lambda
    #  the wave length
    
    def __init__(self, costheta, Lambda):
        self.CosTheta=costheta
        self.Lambda=Lambda
        self.getRayleigh(0)    # to initialize RayleighPar0
        
    ####################################
    ## @brief Rayleigh absorption
    #
    ## @param h
    #  h is the altitude in km
    def getRayleigh(self, h):
        mlambda = self.Lambda/1000. # convert to microns
        q1 = 9.4977E-3
        dum = 1/mlambda**2
        q2 = 0.23465 + 107.6/(146. - dum) + 0.93161/(41.-dum)
        self.RayleighPar0 = q1*q2*q2*dum*dum/self.C/self.HR0
        return self.RayleighPar0*math.exp(-(self.CosTheta*h+self.HTHEMIS)/self.HR0)
        
    ####################################
    ## @brief Rayleigh absorption for ROOT
    #
    ## @param x
    #  x is an array for which x[0] is the altitude in km
    ## @param par
    #  par is here for ROOT.TF1 to be happy    
    
    def getRayleighForTF1(self, x,par):
        h=x[0]
        return self.getRayleigh(h)

        
    ## @brief Rayleigh absorption at sea level
    #
    def getRayleighPar0(self):
        return self.RayleighPar0

if __name__ == '__main__':
    print 'The Rayleigh scattering handler'
    r=pRayleigh(1,532)
    print r.getRayleigh(10)
    print r.getRayleighPar0()
    
    #import ROOT
    #f=ROOT.TF1('f',r.getRayleighForTF1,0.1,25,1)
    #f.Draw()

