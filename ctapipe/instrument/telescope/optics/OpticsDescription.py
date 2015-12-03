import pyhessio as h
from astropy import units as u
import numpy as np
import textwrap
import os

from ctapipe.instrument.util_functions import get_file_type

__all__ = ['initialize']

def initialize(filename,tel_id,item):
    """
    calls the specific initialize function depending on the file
    extension of the given file. The file must already be open/have
    been loaded. The return value of the opening/loading process
    must be given as an argument (item).
    
    Parameters
    ----------
    filename: string
        name of the file
    tel_id: int
        ID of the telescope whose optics information should be loaded
    item: of various type depending on the file extension
        return value of the opening/loading process of the file
    """
    ext = get_file_type(filename)

    function = getattr(Initialize,"_initialize_%s" % ext)
    return function(filename,tel_id,item)
    
    #if 'simtel.gz' in filename:
    #    return _initialize_hessio(filename,tel_id)
    #elif 'fits' in filename:
    #    return _initialize_fits(filename,tel_id,item)

class Initialize:
    
    """`Initialize` is a class containing the initialize functions for
    the different file extensions"""
    
    def _initialize_hessio(filename,tel_id,item):
        """
        reads the Optics data out of the open hessio file
        
        Parameters
        ----------
        filename: string
            name of the hessio file (must be a hessio file!)
        tel_id: int
            ID of the telescope whose optics information should be loaded
        """
        mir_class = -1
        mir_area = h.get_mirror_area(tel_id)*u.m**2
        mir_number = h.get_mirror_number(tel_id)
        prim_mirpar = [-1]
        prim_refrad = -1*u.cm
        prim_diameter = -1*u.cm
        prim_hole_diam = -1*u.cm
        sec_mirpar = [-1]
        sec_refrad = -1*u.cm
        sec_diameter = -1*u.cm
        sec_hole_diam = -1*u.cm
        mir_reflection = [[-1]*u.nm,[-1]]
        opt_foclen = h.get_optical_foclen(tel_id)*u.m
        foc_surfparam = -1
        foc_surf_refrad = -1*u.cm
        tel_trans = -1
    
        return(mir_class,mir_area,mir_number,prim_mirpar,prim_refrad,
               prim_diameter,prim_hole_diam,sec_mirpar,sec_refrad,sec_diameter,
               sec_hole_diam,mir_reflection,opt_foclen,foc_surfparam,
               foc_surf_refrad,tel_trans)
            
    def _initialize_fits(filename,tel_id,item):
        """
        reads the Optics data out of the open fits file
        
        Parameters
        ----------
        filename: string
            name of the fits file (must be a fits file!)
        tel_id: int
            ID of the telescope whose optics information should be loaded
        item: HDUList
            HDUList of the fits file
        """
        hdulist = item
        teles = hdulist[1].data
        telescope_id = teles["TelID"].tolist()
        index = telescope_id.index(tel_id)
        mir_class = -1
        mir_area = hdulist[1].data[index]["MirrorArea"]*u.m**2
        mir_number = hdulist[1].data[index]["NMirrors"]
        prim_mirpar = [-1]
        prim_refrad = -1*u.cm
        prim_diameter = -1*u.cm
        prim_hole_diam = -1*u.cm
        sec_mirpar = [-1]
        sec_refrad = -1*u.cm
        sec_diameter = -1*u.cm
        sec_hole_diam = -1*u.cm
        mir_reflection = [[-1]*u.nm,[-1]]
        opt_foclen = hdulist[1].data[index]["FL"]*u.m
        foc_surfparam = -1
        foc_surf_refrad = -1*u.cm
        tel_trans = -1
                
        return(mir_class,mir_area,mir_number,prim_mirpar,prim_refrad,
               prim_diameter,prim_hole_diam,sec_mirpar,sec_refrad,sec_diameter,
               sec_hole_diam,mir_reflection,opt_foclen,foc_surfparam,
               foc_surf_refrad,tel_trans)
        
    def _initialize_ascii(filename,tel_id,item):
        """
        reads the Optics data out of the ASCII file
        
        Parameters
        ----------
        filename: string
            name of the ASCII file (must be an ASCII config file!)
        tel_id: int
            ID of the telescope whose optics information should be loaded (must
            not be given)
        item: python module
            python module created from an ASCII file using imp.load_source
        """
        dirname = os.path.dirname(filename)
        
        try: mir_class = item.mirror_class[0]
        except: mir_class = -1
                
        try: mir_area = item.mirror_area[0]*u.cm**2
        except: mir_area = -1*u.m**2
        
        try: mir_number = item.mirror_number[0]
        except: mir_number = -1
        
        try: prim_mirpar = item.primary_mirror_parameters
        except: prim_mirpar = [-1]
        
        try: prim_refrad = item.primary_ref_radius[0]*u.cm
        except: prim_refrad = -1*u.cm
        
        try: prim_diameter = item.primary_diameter[0]*u.cm
        except: prim_diameter = -1*u.cm
        
        try: prim_hole_diam = item.primary_hole_diameter[0]*u.cm
        except: prim_hole_diam = -1*u.cm
        
        try: sec_mirpar = item.secondary_mirror_parameters
        except: sec_mirpar = [-1]        
        
        try: sec_refrad = item.secondary_ref_radius[0]*u.cm
        except: sec_refrad = -1*u.cm
        
        try: sec_diameter = item.secondary_diameter[0]*u.cm
        except: sec_diameter = -1*u.cm
        
        try: sec_hole_diam = item.secondary_hole_diameter[0]*u.cm
        except: sec_hole_diam = -1*u.cm
        
        try:
            wavel,reflect = np.loadtxt(dirname+'/'+textwrap.dedent(item.mirror_reflectivity[0]),
                                           unpack=True)
            mir_reflection = [wavel*u.nm,reflect]
        except: mir_reflection = [[-1]*u.nm,[-1]]
        
        try: opt_foclen = item.focal_length[0]*u.cm
        except: opt_foclen = -1*u.m
        
        try: foc_surfparam = item.focal_surface_param
        except: foc_surfparam = [-1]
        
        try: foc_surf_refrad = item.focal_surface_ref_radius[0]*u.cm
        except: foc_surf_refrad = -1*u.cm
        
        try: tel_trans = item.telescope_transmission[0]
        except: tel_trans = -1        
        
        return(mir_class,mir_area,mir_number,prim_mirpar,prim_refrad,
               prim_diameter,prim_hole_diam,sec_mirpar,sec_refrad,sec_diameter,
               sec_hole_diam,mir_reflection,opt_foclen,foc_surfparam,
               foc_surf_refrad,tel_trans)
        
        