#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read an atmospheric profile (Density, P, T...)

@author: Johan Bregeon
"""

ATMOPROF_DICT={'Armazones':26, 'La Palma':36}

def read_atmo_prof(fpath):
    ''' Read atmospheric profile table
    '''
    content=open(fpath).readlines()
    
    altitude=[]
    rho=[]
    thick=[]
    index=[]
    temperature=[]
    pressure=[]
    pw_p=[]
    for line in content:
        if line[0] is '#':
            print(line)
        else:
            ## Alt [km]    rho [g/cm^3] thick [g/cm^2]    n-1        T [K]    p [mbar]      pw / p
            all_el = [float(el) for el in line.split()]
            altitude.append(all_el[0]*1000.)
            rho.append(all_el[1])
            thick.append(all_el[2])
            index.append(all_el[3])
            temperature.append(all_el[4])
            pressure.append(all_el[5])
            pw_p.append(all_el[6])
            
    return altitude, rho, thick, index, temperature, pressure, pw_p
    
    
    
if __name__=="__main__":
    ''' Write up an example of reading a file
    '''
    altitudes, rho, thick, index, temperature, pressure, pw_p = \
       read_atmo_prof(fpath='../ctapipe-extra/ctapipe_resources/atmprof26.dat')