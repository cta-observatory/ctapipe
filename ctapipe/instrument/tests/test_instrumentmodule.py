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

def test_load_file():
    assert(load(filename1))
    assert(load(filename2)==0)
    
def test_read_telescope_data():
    tel1 = Telescope.initialize(filename1,load(filename1))
    tel2 = Telescope.initialize(filename2,load(filename2))
    print(len(tel1[0].tel_id))    
    assert(len(tel1[0].tel_id)>0)
    assert(len(tel2[0].tel_id)>0)
    opt1 = tel1[1]
    opt2 = tel2[1]
    assert(opt1[0].mir_area != 0)    
    assert(opt2[0].mir_area != 0) 
    cam1 = tel1[2]
    cam2 = tel2[2]
    assert(cam1[0].cam_fov != 0)
    assert(cam2[0].cam_fov != 0)
    
def test_read_camera_data():
    cam1 = Camera.initialize(filename1,1,load(filename1))
    cam2 = Camera.initialize(filename2,18,load(filename2))
    assert(cam1.cam_fov != 0)
    assert(cam2.cam_fov != 0)

def test_read_optics_data():
    opt1 = Optics.initialize(filename1,1,load(filename1))
    opt2 = Optics.initialize(filename2,18,load(filename2))
    assert(opt1.mir_area != 0)    
    assert(opt2.mir_area != 0)

def test_close_file():
    assert(close(filename1,load(filename1))==None)
    assert(close(filename2,0)==None)