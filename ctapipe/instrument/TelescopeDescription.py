from astropy import units as u

__all__ = ['get_data']
    
def get_data(instr_table):
    """
    reads the Telescope data out of the instrument table
    
    Parameters
    ----------
    instr_table: astropy table
        name of the astropy table where the whole instrument data read from
        the file is stored
    """
    
    #tel_table,cam_table,opt_table = instr_table
    
    tel_id = [-1]
    tel_num = -1
    tel_posX = [-1]*u.m
    tel_posY = [-1]*u.m
    tel_posZ = [-1]*u.m
    
    for i in range(len(instr_table)):
        try: tel_id = instr_table[i]['TelID']
        except: pass
    
        try: tel_num = len(tel_id)
        except: pass
    
        try:
            tel_posX = instr_table[i]['TelX']
            tel_posY = instr_table[i]['TelY']
            tel_posZ = instr_table[i]['TelZ']
        except: pass
    
    return(tel_id,tel_num,tel_posX,tel_posY,tel_posZ)