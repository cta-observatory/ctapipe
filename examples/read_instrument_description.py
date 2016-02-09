# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:43:50 2015

@author: zornju

Example of using the instrument module and reading data from a hessio, a
fits, and a sim_telarray-config file.
"""

from ctapipe.instrument import InstrumentDescription as ID
from ctapipe.utils.datasets import get_path
import matplotlib.pyplot as plt

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
    print(tel1['1']['TelID'])
    print(tel2['TelescopeTableVersionFeb2016']['TelID'])
    print(tel3['TelescopeTable_CTA-ULTRA6-SCT'])
    print(tel4['TelescopeTableVersionFeb2016']['TelID'])
    print('----------------------------')
    
    #or print all the information stored for a given telescope in a table:
    print('available information about telescope with ID = 1:')
    print(tel1['1'][tel1['1']['TelID']==1])
    print('----------------------------')
    
    #or print a specific information stored for a given telescope in a table:
    print('focal length of telescope with ID = 1:')
    print(tel1['1'][tel1['1']['TelID']==1]['FL'])
    print('----------------------------')


    print('Data from the cfg file stored in the Telescope table:')
    print(tel3['TelescopeTable_CTA-ULTRA6-SCT'])
    print('----------------------------')
    
    #or plot the discriminator pulse shape
    title = 'Discriminator Pulse Shape'
    plt.figure()
    plt.plot(tel3['DiscriminatorPulseShape_PulseShape_MPPC_S10943_Shaped_CutOff350MHz_Prod3']['Time'],\
    tel3['DiscriminatorPulseShape_PulseShape_MPPC_S10943_Shaped_CutOff350MHz_Prod3']['Amplitude'],'+')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
    
    print('Data from the cfg file stored in the Optics table:')
    print(opt3['OpticsTable_CTA-ULTRA6-SCT'])
    print('----------------------------')
    
    #or plot the mirror reflectivity vs. wavelength stored in a
    #config file
    title = 'Mirror reflectivity versus wavelength'
    plt.figure()
    plt.plot(opt3['MirrorRefelctivity_Reflectance_SC-MST_Prod3']['Wavelength'],
             opt3['MirrorRefelctivity_Reflectance_SC-MST_Prod3']['Reflectivity'],
             '+')
    plt.title(title)
    plt.xlabel('Wavelength [%s]' % \
    opt3['MirrorRefelctivity_Reflectance_SC-MST_Prod3']['Wavelength'].unit)
    plt.show()
    
    print('Data from the cfg file stored in the Camera table:')
    print(cam3['CameraTable_CTA-ULTRA6-SCT'])
    print('----------------------------')