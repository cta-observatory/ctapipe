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
    
    #To check which tables are in the dictionaries, do:
    print('Print the name of the tables present in the dictionaries taken',\
    'from the fits file:')
    print(tel1.keys(),cam1.keys(),opt1.keys())
    print('As you can see, in the fits file, only the first dictionary is',\
    'filled with tables.')
    print('------------------------------------------------------------------')
    print('Print the name of the tables present in the dictionaries taken',\
    'from the hessio file:')
    print(tel2.keys(),cam2.keys(),opt2.keys())
    print('------------------------------------------------------------------')
    print('Print the name of the tables present in the dictionaries taken',\
    'from the simtelarray-config file:')
    print(tel3.keys(),cam3.keys(),opt3.keys())
    print('------------------------------------------------------------------')
    print('Print the name of the tables present in the dictionaries taken',\
    'from the faked data:')
    print(tel4.keys(),cam4.keys(),opt4.keys())
    print('------------------------------------------------------------------')
    
    print('Print the table from the telscope dictionary containing the',\
    'overview of the telescope configuration')
    print(tel1['1'])
    print(tel2['TelescopeTableVersionFeb2016'])
    print(tel3['TelescopeTable_CTA-ULTRA6-SCT'])
    print(tel4['TelescopeTableVersionFeb2016'])
    print('------------------------------------------------------------------')
    
    print('Print all the information stored for a given telescope in a table')
    print('available information about telescope with ID = 1:')
    print(tel2['TelescopeTableVersionFeb2016'][tel2['TelescopeTableVersionFeb2016']['TelID']==1])
    print('------------------------------------------------------------------')
    
    print('Print a specific information stored for a given telescope in a',\
    'table, e.g. the x position of the pixels')
    print(cam3['CameraTable_CTA-ULTRA6-SCT']['PixX'])
    print('------------------------------------------------------------------')
    
    print('Plot the discriminator pulse shape')
    title = 'Discriminator Pulse Shape'
    plt.figure()
    plt.plot(tel3['Tel_DiscriminatorPulseShape']['Time'],\
    tel3['Tel_DiscriminatorPulseShape']['Amplitude'],'+')
    plt.title(title)
    plt.xlabel('Time (%s)'% \
    tel3['Tel_DiscriminatorPulseShape']['Time'].unit)
    plt.ylabel('Amplitude')
    plt.show()
    print('------------------------------------------------------------------')
    
    print('Plot the mirror reflectivity vs wavelength stored in a config file')
    title = 'Mirror reflectivity versus wavelength'
    plt.figure()
    plt.plot(opt3['Opt_MirrorRefelctivity']['Wavelength'],
             opt3['Opt_MirrorRefelctivity']['Reflectivity'],
             '+')
    plt.title(title)
    plt.xlabel('Wavelength (%s)' % \
    opt3['Opt_MirrorRefelctivity']['Wavelength'].unit)
    plt.show()