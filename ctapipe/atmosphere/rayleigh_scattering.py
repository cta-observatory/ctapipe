#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extinction Profile Volume Scattering for standard air condition

A.Bucholtz Applied Optics vol. 34 n. 15 p. 2769 - 20/05/1995 - Table 3

BetaS refers to Beta for fPs=1013.25 mbar and fTs=288.15 K

Rayleigh Volume-Scattering Coefficient \\beta_s in km^-1:
Verification BetaS(0.35) = 7.4473e-02 ~ 7.450e-02 in Table 2.
Verification BetaS(0.53) = 1.3352e-02 ~ 1.336e-02 in Table 2.\n
-->Converted to m^-1

@author: Johan Bregeon (LUPM)
"""

from math import pow


def rayleigh_extinction_PT(wl=355):
    ''' Compute Rayleigh scattering BetaS at reference pressure and temperature
    '''
    #print("Calculating Reference Extinction for wl=",wl)  
    A=0.
    B=0.
    C=0.
    D=0.  
    if wl>0.2 and wl<=0.5:
        A = 7.68246*1e-4
        B = 3.55212
        C = 1.35579
        D = 0.11563
    elif wl>0.5 and wl<2.2:
        A = 10.21675*1e-4
        B = 3.99668
        C = 1.10298*1e-3
        D = 2.71393*1e-2
    else:
      print("Wave length not in range 0.2 - 2.2 um")
      return -1
    # converted to m^-1
    betaS = A * pow( wl, -(B + C*wl + D/wl) ) / 1000.
    return betaS;


def rayleigh_extinction(wl, P, T):
    ''' Compute Rayleigh scattering extinction for a given wave lenght
    temperature and pressure
    '''    
    fPs = 1013.25 #  standard pressure in mbar
    fTs = 288.15  #  standard temerature in K (15°C)    
    betaS = rayleigh_extinction_PT(wl)
    beta = betaS * (P/fPs) * (fTs/T)    
    return beta


if __name__ == '__main__':

    print('Verification BetaS(0.35) = 7.4473e-02 ~ 7.450e-02')
    print('Rayleigh scattering extinction for wl=355 : ',
              rayleigh_extinction_PT(0.35))
    print()
    print('Verification BetaS(0.53) = 1.3352e-02 ~ 1.336e-02')    
    print('Rayleigh scattering extinction for wl=532 : ',
              rayleigh_extinction_PT(0.53))

    print('Rayleigh scattering extinction for wl=400, P=1000, T=250 : ',
              rayleigh_extinction(0.4, 1000, 250))