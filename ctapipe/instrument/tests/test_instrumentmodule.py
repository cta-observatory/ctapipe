# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 17:37:19 2015

@author: zornju
"""

import ctapipe.instrument.InstrumentDescription as ID
from ctapipe.utils.datasets import get_path
import os

filename1 = get_path('PROD2_telconfig.fits.gz')
filename2 = get_path('gamma_test.simtel.gz')
filename3 = get_path('CTA-ULTRA6-SCT.cfg')

def test_load_and_write_telescope_data():
    tel1,cam1,opt1 = ID.load(filename1)
    tel2,cam2,opt2 = ID.load(filename2)
    tel3,cam3,opt3 = ID.load(filename3)
    tel4,cam4,opt4 = ID.load()
    assert(len(tel1)>0)
    assert(len(tel2)>0)
    assert(len(opt3)>0)
    assert(len(tel4)>0)
    
    ID.write_fits(instr_dict=tel1,filename='test.fits',overwrite=True)
    tel11,cam11,opt11 = ID.load('test.fits')
    assert(len(tel11)>0)
    os.remove('test.fits')