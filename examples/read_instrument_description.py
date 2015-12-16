# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:43:50 2015

@author: zornju

Example of using the instrument module and reading data from a hessio, a
fits, and a sim_telarray-config file.
"""

from ctapipe.instrument import InstrumentDescription as ID, CameraDescription as CD
from ctapipe.utils.datasets import get_path
from astropy import units as u
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    filename1 = get_path('PROD2_telconfig.fits.gz')
    filename2 = get_path('gamma_test.simtel.gz')
    #filename3 = get_path('CTA-ULTRA6-SCT.cfg')
    filename3 = '/afs/ifh.de/group/cta/MC_Production/d20150828_GM/mrg/sim_telarray/cfg/CTA/CTA-ULTRA6-SCT.cfg'    
    
    instr1 = ID.Telescope.from_file(filename1)
    instr2 = ID.Telescope.from_file(filename2)
    instr3 = ID.Telescope.from_file(filename3)

    #The ID.telescope.initializ-function returns 3 objects as a list. The
    #first entry of the list is the object containing the telescope
    #configuration whithout the camera and optics, thus:
    tel1 = instr1[0]
    tel2 = instr2[0]
    tel3 = instr3[0]

    #The second entry of the list is the object containing the configuration
    #of the telescope optics, thus:
    opt1 = instr1[1]     
    opt2 = instr2[1]
    opt3 = instr3[1]
    
    #The third entry of the list is the object containing the configuration
    #of the camera, thus:
    cam1 = instr1[2]
    cam2 = instr2[2]
    cam3 = instr3[2]
    
    #Now we can print some telescope, optics and camera configurations, e.g.
    #the telescoppe IDs of all telescopes whose data is stored in the files
    print('Telescope IDs (of fits and hessio file):')
    print(tel1.tel_id)
    print(tel2.tel_id)
    print('----------------------------')
    
    #and the mirror areas of the first telescope of the files (if the data of
    #a specific configuration is not stored in the file, a `-1` is retured.)
    print('Mirror Area of the first telescope (of fits and hessio file).',
          'Returned, when the whole telescope data is read from the file:')
    print('{0:.2f}'.format(opt1[0].mir_area))
    print('{0:.2f}'.format(opt2[0].mir_area))
    print('Primary mirror parameters of ASCII file:')
    print(opt3[0].prim_mirpar)
    print('----------------------------')
    
    #and the camera FOV of the second telescope of the files:
    print('Camera FOV of the second telescope (of fits, hessio, and ASCII file).',
          'Returned, when the whole telescope data is read from the file:')
    print('{0:.2f}'.format(cam1[1].cam_fov))
    print('{0:.2f}'.format(cam2[1].cam_fov))
    print('{0:.2f}'.format(cam3[0].cam_fov))
    print('----------------------------')
    
    #But, one can also load only the camera or the optics configuration of
    #a specific telescope with ID `tel_id` (of course the configuration of this
    #specific telescope must be stored in the file) without loading all the
    #telescope configurations, e.g. load camera and optics configurations of
    #camera with ID 1 from FITS and ASCII file:
    cam10 = ID.Camera.from_file(filename1,1)
    cam30 = ID.Camera.from_file(filename3,1)
    print('Camera FOV of the first telescope of fits file.',
          'Returned, when only the camera data is read from the file:')
    print('{0:.2f}'.format(cam10.cam_fov))
    print('Rotate the Camera by 180 degree, i.e. the x- & y-positions will be',
          'rotated by 190 degree:')
    print('x-position of the camera before using the rotate method:')
    print(cam10.pix_posX)
    print('x-position of the camera after using the rotate method:')
    cam10.rotate(cam10,180*u.degree)
    print(cam10.pix_posX)
    print('FADC pulse shape over time of ASCII file')
    plt.plot(cam30.fadc_pulsshape[0],cam30.fadc_pulsshape[1],'+')
    plt.show()    
    print('----------------------------')
    
    print('Print imported camera data as a table:')
    table = CD.write_table(1,cam10.cam_class,cam10.pix_id[:1],cam10.pix_posX,
                           cam10.pix_posY,cam10.pix_area,cam10.pix_type)
    print(table)
    print('---------------------------')
    
    opt10 = ID.Optics.from_file(filename1,1)
    opt30 = ID.Optics.from_file(filename3,1)
    print('Mirror Area of the first telescope of fits file.',
          'Returned, when only the optics data is read from the file:')
    print('{0:.2f}'.format(opt10.mir_area))
    print('Mirror reflection over wavelength [nm]')
    plt.plot(opt30.mir_reflection[0].value,opt30.mir_reflection[1],'+')
    plt.show()
    print('----------------------------')