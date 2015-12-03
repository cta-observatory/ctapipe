import pyhessio as h
from astropy import units as u

from ctapipe.instrument.util_functions import get_file_type

__all__ = ['initialize']

def initialize(filename,item):
    """
    calls the specific initialize function depending on the file
    extension of the given file. The file must already be open/have
    been loaded. The return value of the opening/loading process
    must be given as an argument (item).
    
    Parameters
    ----------
    filename: string
        name of the file
    item: of various type depending on the file extension
        return value of the opening/loading process of the file
    """
    ext = get_file_type(filename)

    function = getattr(Initialize,"_initialize_%s" % ext)
    return function(filename,item)
    
    #if 'simtel.gz' in filename:
    #   return _initialize_hessio(filename)
    #elif 'fits' in filename:
    #    return _initialize_fits(filename,item)
        
class Initialize:
    
    """`Initialize` is a class containing the initialize functions for
    the different file extensions"""
    
    def _initialize_hessio(filename,item):
        """
        reads the Telescope data out of the open hessio file
        
        Parameters
        ----------
        filename: string
            name of the hessio file (must be a hessio file!)
        """
        #tel_id = h.get_telescope_ids()  #--> this function only can be used if the
                                        #according python wrapper hase been added
                                        #to pyhessio.c and hessio.py
        
        tel_id = h.get_teldata_list()
        tel_num = h.get_num_telescope()
        tel_posX = [-1]*u.m
        tel_posY = [-1]*u.m
        tel_posZ = [-1]*u.m
        
        return(tel_id,tel_num,tel_posX,tel_posY,tel_posZ)
    
    def _initialize_fits(filename,item):
        """
        reads the Telescope data out of the open fits file
        
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
        tel_id = teles["TelID"]
        tel_num = len(tel_id)
        tel_posX = teles["TelX"]*u.m
        tel_posY = teles["TelY"]*u.m
        tel_posZ = teles["TelZ"]*u.m
        
        return(tel_id,tel_num,tel_posX,tel_posY,tel_posZ)
    
    def _initialize_ascii(filename,item):
        """
        reads the Telescope data out of the ASCII file
        
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
        tel_id = [-1]
        tel_num = -1
        tel_posX = [-1]*u.m
        tel_posY = [-1]*u.m
        tel_posZ = [-1]*u.m
        
        return(tel_id,tel_num,tel_posX,tel_posY,tel_posZ)
