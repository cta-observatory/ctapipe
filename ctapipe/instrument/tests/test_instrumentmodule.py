# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 17:37:19 2015

@author: zornju
"""

import ctapipe.instrument.InstrumentDescription as ID
import ctapipe.instrument.CameraDescription as CD
from ctapipe.utils.datasets import get_path
import os
import numpy as np
import astropy.units as u

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
    
    ID.write_fits(instr_dict=tel4,filename='test.fits',overwrite=True)
    tel41,cam41,opt41 = ID.load('test.fits')
    assert(len(tel41)>0)
    os.remove('test.fits')

def test_make_rectangular_camera_geometry():
    geom = CD.make_rectangular_camera_geometry()
    assert(geom.pix_X.shape == geom.pix_Y.shape)


def test_rotate_camera():
    geom = CD.make_rectangular_camera_geometry(10, 10)
    geom.rotate(geom,'10d')


def test_guess_camera():
    px = np.linspace(-10, 10, 11328) * u.m
    py = np.linspace(-10, 10, 11328) * u.m
    geom = CD.Camera.guess(px, py)
    assert geom.pix_type.startswith('rect')


def test_find_neighbor_pixels():
    x, y = np.meshgrid(np.linspace(-5, 5, 5), np.linspace(-5, 5, 5))
    neigh = CD.find_neighbor_pixels(x.ravel(), y.ravel(), rad=3.1)
    assert(set(neigh[11]) == set([16, 6, 10, 12]))

