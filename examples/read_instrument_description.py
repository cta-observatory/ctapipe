# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:43:50 2015

@author: zornju

Example of using the instrument module and reading data from a hessio and a
fits file.
"""

from ctapipe.instrument import InstrumentDescription as ID,util_functions as uf
from ctapipe.utils.datasets import get_path

if __name__ == '__main__':
    
    filename1 = get_path('PROD2_telconfig.fits.gz')
    filename2 = get_path('gamma_test.simtel.gz')
    
    #open both files:
    item1 = uf.load(filename1)
    item2 = uf.load(filename2)
    
    #initialize the whole telescope, i.e. read all data from the files which
    #concerns the whole telescope (including camera and optics)
    instr1 = ID.Telescope.initialize(filename1,item1)
    instr2 = ID.Telescope.initialize(filename2,item2)

    #The ID.telescope.initializ-function returns 3 objects as a list. The
    #first entry of the list is the object containing the telescope
    #configuration whithout the camera and optics, thus:
    tel1 = instr1[0]
    tel2 = instr2[0]

    #The second entry of the list is the object containing the configuration
    #of the telescope optics, thus:
    opt1 = instr1[1]     
    opt2 = instr2[1]
    
    #The third entry of the list is the object containing the configuration
    #of the camera, thus:
    cam1 = instr1[2]
    cam2 = instr2[2]
    
    #Now we can print some telescope, optics and camera configurations, e.g.
    #the telescoppe IDs of all telescopes whose data is stored in the files
    print(tel1.tel_id)
    print(tel2.tel_id)
    
    #and the mirror areas of the first telescope of the files (if the data of
    #a specific configuration is not stored in the file, a `-1` is retured.)
    print(opt1[0].mir_area)
    print(opt2[0].mir_area)
    
    #and the camera FOV of the second telescope of the files:
    print(cam1[1].cam_fov)
    print(cam2[1].cam_fov)
    
    #But, one can also load only the camera or the optics configuration of
    #a specific telescope with ID `tel_id` (of course the configuration of this
    #specific telescope must be stored in the file) without loading all the
    #telescope configurations, e.g. load camera and optics configurations of
    #camera with ID 1 and ID 18:
    cam1 = ID.Camera.initialize(filename1,1,item1)
    cam2 = ID.Camera.initialize(filename2,18,item2)
    print(cam1.cam_fov)
    print(cam2.cam_fov)
    
    opt1 = ID.Optics.initialize(filename1,1,item1)
    opt2 = ID.Optics.initialize(filename2,18,item2)
    print(opt1.mir_area)
    print(opt2.mir_area)
    
    #At the end, the files should be closed:
    uf.close(filename1,item1)
    uf.close(filename2,item2)