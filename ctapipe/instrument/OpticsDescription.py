from astropy import units as u

__all__ = ['get_data']

def get_data(instr_table,tel_id):
    """
    reads the Optics data out of the instrument table
    
    Parameters
    ----------
   instr_table: astropy table
        name of the astropy table where the whole instrument data read from
        the file is stored
    tel_id: int
        ID of the telescope whose optics information should be loaded
    """
    
    mir_class = -1
    mir_area = -1*u.m**2
    mir_number = -1
    prim_mirpar = [-1]
    prim_refrad = -1*u.cm
    prim_diameter = -1*u.cm
    prim_hole_diam = -1*u.cm
    sec_mirpar = [-1]
    sec_refrad = -1*u.cm
    sec_diameter = -1*u.cm
    sec_hole_diam = -1*u.cm
    mir_reflection = [[-1]*u.nm,[-1]]
    opt_foclen = -1*u.m
    foc_surfparam = -1
    foc_surf_refrad = -1*u.cm
    tel_trans = -1  
        
    #tel_table,cam_table,opt_table = instr_table
    
    for i in range(len(instr_table)):
    
        try: tel_id_bool = (instr_table[i]['TelID']==tel_id)
        except: pass
        
        try: mir_class = instr_table[i][tel_id_bool]['MirClass']
        except: pass
    
        try: mir_area = instr_table[i][tel_id_bool]['MirrorArea']
        except: pass
    
        try: mir_number = instr_table[i][tel_id_bool]['NMirrors']
        except: pass
    
        try: opt_foclen = instr_table[i][tel_id_bool]['FL']
        except: pass    
        #except TypeError:
        #    unit = raw_input('The unit of the optical focal length is not given. Please enter it here: ')
    

    return(mir_class,mir_area,mir_number,prim_mirpar,prim_refrad,
           prim_diameter,prim_hole_diam,sec_mirpar,sec_refrad,sec_diameter,
           sec_hole_diam,mir_reflection,opt_foclen,foc_surfparam,
           foc_surf_refrad,tel_trans)   