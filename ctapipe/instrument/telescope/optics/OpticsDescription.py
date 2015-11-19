import hessio as h
from astropy import units as u

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
        mir_area = h.get_mirror_area(tel_id)*u.m**2
        mir_number = h.get_mirror_number(tel_id)
        opt_foclen = h.get_optical_foclen(tel_id)*u.m
    
        return mir_area,mir_number,opt_foclen
            
    def _initialize_fits(filename,tel_id,item):
        """
        reads the Optics data out of the open fits file
        
        Parameters
        ----------
        filename: string
            name of the hessio file (must be a fits file!)
        tel_id: int
            ID of the telescope whose optics information should be loaded
        item: HDUList
            HDUList of the fits file
        """
        hdulist = item
        teles = hdulist[1].data
        telescope_id = teles["TelID"].tolist()
        index = telescope_id.index(tel_id)
        mir_area = hdulist[1].data[index]["MirrorArea"]*u.m**2
        mir_number = hdulist[1].data[index]["NMirrors"]
        opt_foclen = hdulist[1].data[index]["FL"]*u.m
        
        return mir_area,mir_number,opt_foclen 
