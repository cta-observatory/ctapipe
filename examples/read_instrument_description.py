# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:43:50 2015

@author: zornju

Example of using the instrument module and reading data from a hessio, a
fits, and a sim_telarray-config file.
"""

from ctapipe.instrument import InstrumentDescription as ID
from ctapipe.utils.datasets import get_path

if __name__ == '__main__':
    
    filename1 = get_path('PROD2_telconfig.fits.gz')
    filename2 = get_path('gamma_test.simtel.gz')
    filename3 = get_path('CTA-ULTRA6-SCT.cfg')
    
    tel1,cam1,opt1 = ID.load(filename1)
    tel2,cam2,opt2 = ID.load(filename2)
    tel3,cam3,opt3 = ID.load(filename3)
    tel4,cam4,opt4 = ID.load()
    
    #Now we can print some telescope, optics and camera configurations, e.g.
    #the telescoppe IDs of all telescopes whose data is stored in the files
    print('Some of the tables which were read from the files:')
    print(tel1['1'])
    print(tel2['TelescopeTableVersionFeb2016'])
    print(opt3['OpticsTable_CTA-ULTRA6-SCT'])
    print(tel4['TelescopeTableVersionFeb2016'])
    print('----------------------------')