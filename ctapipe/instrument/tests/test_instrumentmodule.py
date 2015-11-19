# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 17:37:19 2015

@author: zornju
"""

from ctapipe.instrument.InstrumentDescription import Telescope, Camera, Optics
from ctapipe.instrument.util_functions import load, close
from ctapipe.utils.datasets import get_path

filename1 = get_path('PROD2_telconfig.fits.gz')
filename2 = get_path('gamma_test.simtel.gz')

item1 = load(filename1)
item2 = load(filename2)

def test_load_file():
    assert(item1)
    assert(item2==0)
    
def test_read_telescope_data():
    instr1 = Telescope.initialize(filename1,item1)
    instr2 = Telescope.initialize(filename2,item2)
    tel1 = instr1[0]
    tel2 = instr2[0]    
    assert(len(tel1.tel_id)>0)
    assert(len(tel2.tel_id)>0)
    opt1 = instr1[1]
    opt2 = instr2[1]
    assert(opt1[0].mir_area != 0)    
    assert(opt2[0].mir_area != 0) 
    cam1 = instr1[2]
    cam2 = instr2[2]
    assert(cam1[0].cam_fov != 0)
    assert(cam2[0].cam_fov != 0)
    
def test_read_camera_data():
    cam1 = Camera.initialize(filename1,1,item1)
    cam2 = Camera.initialize(filename2,18,item2)
    assert(cam1.cam_fov != 0)
    assert(cam2.cam_fov != 0)

def test_read_optics_data():
    opt1 = Optics.initialize(filename1,1,item1)
    opt2 = Optics.initialize(filename2,18,item2)
    assert(opt1.mir_area != 0)    
    assert(opt2.mir_area != 0)

def test_close_file():
    assert(close(filename1,item1)==None)
    assert(close(filename2,item2)==None)