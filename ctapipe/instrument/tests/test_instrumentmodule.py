# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 17:37:19 2015

@author: zornju
"""

import os

import astropy.units as u
import ctapipe.instrument.InstrumentDescription as ID
import numpy as np
from ctapipe.instrument import CameraGeometry
from ctapipe.utils import get_dataset
import pytest

filename1 = get_dataset('PROD2_telconfig.fits.gz')
filename2 = get_dataset('gamma_test.simtel.gz')
filename3 = get_dataset('CTA-ULTRA6-SCT.cfg')

@pytest.mark.skip(reason="instrument module currently broken")
def test_load_and_write_telescope_data():
    tel1,cam1,opt1 = ID.load(filename1)
    tel2,cam2,opt2 = ID.load(filename2)
    tel3,cam3,opt3 = ID.load(filename3)
    tel4,cam4,opt4 = ID.load()
    assert(len(tel1)>0)
    assert(len(tel2)>0)
    assert(len(opt3)>0)
    assert(len(tel4)>0)
    
    ID.write_fits(instr_dict=tel4,filename='test.fits',overwrite=True)
    tel41,cam41,opt41 = ID.load('test.fits')
    assert(len(tel41)>0)
    os.remove('test.fits')

