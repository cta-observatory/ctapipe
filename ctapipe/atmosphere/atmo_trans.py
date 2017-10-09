#!/bin/env python

# Read the atmospheric transmission and extinction coefficient
# for a given Modtran like atmospheric transmission table
#

import numpy
import os
from scipy import interpolate

def readAtmoTrans(fname='../data/atm_trans_1800_1_10_0_0_1800.dat'):
    ''' Read atmospheric transmission table
    '''
    content=open(absName).readlines()

    opt_depth={}
    extinction={}

    for l in content[14:]:
        all=l.split()
        if l[0]=='#':
            obs_alt=float(all[2].strip(','))
            altitudes=numpy.array(all[4:],'f') #a.s.l.
            altitudes=altitudes*1000 # in meters
        else:
            opt_depth[float(all[0])]=numpy.array(all[1:],'f')

    for wl,depth in opt_depth.items():
        alpha=[]
        for i in range(len(depth)-1):
            alpha.append((depth[i+1]-depth[i])/(altitudes[i+1]-altitudes[i]))
        extinction[wl]=numpy.array(alpha,'f')

    return content[:15], obs_alt, altitudes, opt_depth, extinction


def writeAtmoTrans(run, header, scaled_od, fname=None):
    ''' Dump the scaled optical depth table to an ASCII file
    in Kaskade/Corsika input file format
    '''
    if fname is None:
        fname='atm_trans_Lidar_%s.dat'%run
    content=header
    for wl, od_prof in scaled_od.items():
        txt='        %03d     '%wl
        for od in od_prof:
            if od<10:
                txt+='%10.6f'%od
            else:
                txt+='%10.2f'%od
        content.append(txt+'\n')
    open(fname,'w').writelines(content)
    print('Done.')


def getODForAltitudes(npalt, npod, newalt, opt_depth_wl):
    ''' Get Transmission values for a given array of altitudes
    '''
    # x=alt y=nptrans
    tck = interpolate.splrep(npalt, npod, s=0)
    newod = interpolate.splev(newalt, tck, der=0)
    # no extrapolation, use model outside measurements
    deltaod=0
    for i in range(len(newalt)):
        #print('%s alt %s %s %s'%(i, newalt[i], npalt[0], npalt[-1]))
        if newalt[i]<npalt[0]: #below AltMin - constant
            newod[i]=npod[0]
        elif newalt[i]>npalt[-2]: #above R0 -1 bin - follow model
            if deltaod==0:
                deltaod=newod[i]-opt_depth_wl[i]
                #print('deltaod %s %s %s'%(deltaod, i, newalt[i]))
            newod[i]=opt_depth_wl[i]+deltaod
    return newod

def getODRatio(newod, opt_depth_wl):
    od_ratio=newod/opt_depth_wl
    return od_ratio

def getScaledOD(opt_depth, od_ratio, corr=1):
    ''' Scale all wavelength using 532 nm measurement
    '''
    scaled_od={}
    for wl,od in opt_depth.items():
        scaled_od[wl]=opt_depth[wl]*od_ratio*corr #pow(wl/532.,1.3)
        for i in range(len(scaled_od[wl])):
            if scaled_od[wl][i]>10:
                scaled_od[wl][i]=99999.00
    return scaled_od

def getIndex(myarray,val):
    return len(myarray)-sum(myarray>val)

if __name__=="__main__":
    ''' Write up an example of reading
    '''
