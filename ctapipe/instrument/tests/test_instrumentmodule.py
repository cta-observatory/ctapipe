# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 17:37:19 2015

@author: zornju
"""

import ctapipe.instrument.InstrumentDescription as ID
from ctapipe.utils.datasets import get_path

filename1 = get_path('PROD2_telconfig.fits.gz')
filename2 = get_path('gamma_test.simtel.gz')
filename3 = get_path('CTA-ULTRA6-SCT.cfg')

def test_read_telescope_data():
    instr1 = ID.Telescope.from_file(filename1)
    instr2 = ID.Telescope.from_file(filename2)
    instr3 = ID.Telescope.from_file(filename3)
    instr4 = ID.Telescope.from_file()
    tel1 = instr1[0]
    tel2 = instr2[0]
    tel3 = instr3[0]
    tel4 = instr4[0]
    assert(len(tel1.tel_id)>0)
    assert(len(tel2.tel_id)>0)
    assert(len(tel3.tel_id)>0)
    assert(len(tel4.tel_id)>0)
    opt1 = instr1[1]
    opt2 = instr2[1]
    opt3 = instr3[1]
    opt4 = instr4[1]
    assert(opt1[0].mir_area.value != 0)    
    assert(opt2[0].mir_area.value != 0)
    assert(opt3[0].mir_area.value != 0)
    assert(opt4[0].mir_area.value != 0)
    cam1 = instr1[2]
    cam2 = instr2[2]
    cam3 = instr3[2]
    cam4 = instr4[2]
    assert(cam1[0].cam_fov.value != 0)
    assert(cam2[0].cam_fov.value != 0)
    assert(cam3[0].cam_fov.value != 0)
    assert(cam4[0].cam_fov.value != 0)