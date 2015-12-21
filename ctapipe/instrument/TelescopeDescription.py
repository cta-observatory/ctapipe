import pyhessio as h
from astropy import units as u

__all__ = ['from_file_hessio','from_file_fits','from_file_ascii']
    
def from_file_hessio(filename,item):
    """
    reads the Telescope data out of the open hessio file
    
    Parameters
    ----------
    filename: string
        name of the hessio file (must be a hessio file!)
    """
    tel_id = h.get_telescope_ids()
    tel_num = h.get_num_telescope()
    tel_posX = [-1]*u.m
    tel_posY = [-1]*u.m
    tel_posZ = [-1]*u.m
    
    return(tel_id,tel_num,tel_posX,tel_posY,tel_posZ)

def from_file_fits(filename,item):
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
    
    tel_id = [-1]
    tel_num = -1
    tel_posX = [-1]*u.m
    tel_posY = [-1]*u.m
    tel_posZ = [-1]*u.m
    
    hdulist = item
    for i in range(len(hdulist)):
        teles = hdulist[i].data
        
        try: tel_id = teles["TelID"]
        except: pass
    
        try: tel_num = len(tel_id)
        except: pass
    
        try: tel_posX = teles["TelX"]*u.m
        except: pass
    
        try: tel_posY = teles["TelY"]*u.m
        except: pass
    
        try: tel_posZ = teles["TelZ"]*u.m
        except: pass
    
    return(tel_id,tel_num,tel_posX,tel_posY,tel_posZ)

def from_file_ascii(filename,item):
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
