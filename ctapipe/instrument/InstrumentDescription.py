import numpy as np
from ctapipe.instrument import CameraDescription as CD
from ctapipe.io.files import get_file_type
from astropy import units as u
import pyhessio as h
import os
from astropy.table import Table
import random
import imp
import textwrap

__all__ = ['load','load_fakedata','load_hessio','load_fits','load_config',
           'read_config_data','get_var_from_file','write_fits']

#class Optics:
    
#    """`Optics` is a class that provides and gets all the information about
#    the optics of a specific telescope."""


    
#class Telescope(Optics,Camera):
    
#    """`Telescope` is a class that provides and gets all the information about
#    all telescopes available in a run. It inherits the methods and variables of
#    the classes `Optics` and `Camera`."""


class Atmosphere:
    """Atmosphere is a class that provides data about the atmosphere. This
    data is stored in different files which are read by member functions"""
    def __init__(self,rho,thickness,ext_coeff):
        self.rho = rho
        self.thickness = thickness
        self.ext_coeff = ext_coeff
    
    def load_profile(filename):
        """
        Load atmosphere profile from file
        
        Parameter
        ---------
        filename: string
            name of file
        --------
        """
        altitude,rho,thickness,n_minus_1 = np.loadtxt(filename,unpack=True,
                                                      delimeter=' ')
        altitude = altitude*u.km
        rho = rho*u.g*u.cm**-3
        thickness = thickness*u.g*u.cm**-2
        return altitude,rho,thickness
    
    def load_extinction_coeff(filename):
        """
        Load atmosphere extinction profile from file
        
        Parameter
        ---------
        filename: string
            name of file
        --------
        """
        
        # still work to do
           

def load(filename = '', path = None,version = '',instr_item = '',telID = ''):
    """
    Function to load instrument data from a file
    
    Parameters
    ----------
    filename: string
        name of the file
    path: string
        name of the path, where the file(s) are located
        - when already given as part of the filename, nothing to be entered
        - when file(s) is located in current directory, nothing to be entered
    version: string
        needed if no filename is given and if files with a given version should
        be looked for
    instr_item: string
        item of the instrument (Telescope,Camera or Optics) whose data
        should be loaded, can be either a list, e.g. ('Camera','Telescope') or
        just one string, e.g. 'Camera'
    Return
    ------
    possibility 1:
    telescope,camera,optics = load_hessio(filename)
    
    possibility2:
    telescope,camera,optics = load_fits(filename = '',path = None,
                                        version = '',instr_item = '')
    
    all dictionaries contain astropy.table Tables
    """
    filetype = get_file_type(filename)
    
    if filename == '' and version == '':
        return load_fakedata()
    elif filetype == 'fits' or version != '' :
        return load_fits(filename)
    elif filetype == 'cfg':
        return load_config(filename)
    elif filetype == 'simtel':
        return load_hessio(filename)
    else:
        raise TypeError("File type {} not supported".format(filetype))

def load_fakedata():
    """
    Function writing faked date into an astropy.table Table
    
    Return
    ------
    telescope,camera,optics: 3 dictionaries
        all dictionaries contain astropy.table Tables
    """
    print('Faked data will be produced.')
    version = 'Feb2016'
    tel_num = 10
    tel_id = [random.randrange(1,124,1) for x in range(0,tel_num)]
    
    telescope = {}
    camera = {}
    optics = {}  
    
    tel_table_prime = Table()
    tel_table_prime['TelID']= tel_id
    
    tel_posX = [random.uniform(1,100) for x in range(tel_num)]
    tel_posY = [random.uniform(1,100) for x in range(tel_num)]
    tel_posZ = [random.uniform(1,100) for x in range(tel_num)]
    tel_table_prime['TelX'] = tel_posX
    tel_table_prime['TelX'].unit = u.m
    tel_table_prime['TelY'] = tel_posY
    tel_table_prime['TelY'].unit = u.m
    tel_table_prime['TelZ'] = tel_posZ
    tel_table_prime['TelZ'].unit = u.m
    mirror_area = [random.uniform(1,100) for x in range(tel_num)]
    tel_table_prime['MirA'] = mirror_area
    tel_table_prime['MirA'].unit = u.m**2
    mirror_num = [random.randrange(1,124,1) for x in range(tel_num)]
    tel_table_prime['MirN'] = mirror_num
    foclen = [random.uniform(1,100) for x in range(tel_num)]
    tel_table_prime['FL'] = foclen
    tel_table_prime['FL'].unit = u.m
    
    telescope['TelescopeTableVersion%s' % version] = tel_table_prime    
    
    for t in range(len(tel_id)):       
        
        cam_table_prime = Table()
        opt_table_prime = Table()
        pixel_num = 128
        pix_posX = [random.uniform(1,100) for x in range(tel_num*pixel_num)]
        pix_posY = [random.uniform(1,100) for x in range(tel_num*pixel_num)]       
        pix_id = np.arange(len(pix_posX))
        pix_area = [random.uniform(1,100) for x in range(tel_num*pixel_num)]
         
        cam_table_prime.meta = {'TELID': tel_id[t], 'VERSION': version, \
        'PIXX_DES': 'x-position of the pixel measured by...'}
        cam_table_prime['PixID'] = pix_id
        cam_table_prime['PixX'] = pix_posX
        cam_table_prime['PixX'].unit = u.m
        cam_table_prime['PixY'] = pix_posY
        cam_table_prime['PixY'].unit = u.m
        cam_table_prime['PixA'] = pix_area
        cam_table_prime['PixA'].unit = u.mm**2
        
        opt_table_prime.meta = {'TELID': tel_id[t], 'VERSION': version, \
        'MIRN': mirror_num[t], 'MIRA': mirror_area[t]*u.m**2, \
        'FL': foclen[t]*u.m, \
        'MIRA_DES': 'Area of all mirrors'}
        tab_mirrefl = Table()
        tab_mirrefl['wavel'] = np.arange(100,700,10)
        tab_mirrefl['wavel'].unit = u.nm
        tab_mirrefl['refl'] = [random.uniform(0.01,1) \
        for x in range(len(tab_mirrefl['wavel']))]
        
        opt_table_prime['MirR'] = tab_mirrefl
        
        camera['CameraTable_Version%s_TelID%i' % (version,tel_id[t])] \
        = cam_table_prime
        optics['OpticsTable_Version%s_TelID%i' % (version,tel_id[t])] \
        = opt_table_prime
    print('Astropy tables have been created.')
    return telescope,camera,optics
    
def load_hessio(filename):
    """
    Function to open and load a hessio file
    
    Parameters
    ----------
    filename: string
        name of the file
    """
    print("Hessio file will be opened.")    
    h.file_open(filename)
    next(h.move_to_next_event())
    #version = h.get...
    version = 'Feb2016'
    tel_id = h.get_telescope_ids()
    
    telescope = {}
    camera = {}
    optics = {}  
    
    tel_table_prime = Table()
    tel_table_prime['TelID']= tel_id
    
    tel_posX = [h.get_telescope_position(i)[0] for i in tel_id]
    tel_posY = [h.get_telescope_position(i)[1] for i in tel_id]
    tel_posZ = [h.get_telescope_position(i)[2] for i in tel_id]
    tel_table_prime['TelX'] = tel_posX
    tel_table_prime['TelX'].unit = u.m
    tel_table_prime['TelY'] = tel_posY
    tel_table_prime['TelY'].unit = u.m
    tel_table_prime['TelZ'] = tel_posZ
    tel_table_prime['TelZ'].unit = u.m
    #tel_table['CameraClass'] = [h.get_camera_class(i) for i in tel_id]
    tel_table_prime['MirA'] = [h.get_mirror_area(i) for i in tel_id]
    tel_table_prime['MirA'].unit = u.m**2
    tel_table_prime['MirN'] = [h.get_mirror_number(i) for i in tel_id]
    tel_table_prime['FL'] = [h.get_optical_foclen(i) for i in tel_id]
    tel_table_prime['FL'].unit = u.m
    
    tel_table_prime.meta = {'VERSION': version, 'TELNUM': len(tel_posX),\
    'TELX_DES': 'x-position of the pixel measured by...'}
    
    telescope['TelescopeTableVersion%s' % version] = tel_table_prime
    
    
    for t in range(len(tel_id)):       
        
        cam_table_prime = Table()
        opt_table_prime = Table()
        pix_posX = h.get_pixel_position(tel_id[t])[0]
        pix_posY = h.get_pixel_position(tel_id[t])[1]
        pix_id = np.arange(len(pix_posX))        
    
        try:
            camera_class = CD.guess_camera_geometry(pix_posX*u.m,pix_posY*u.m)
            pix_area_prime = camera_class.pix_area
            pix_type_prime = camera_class.pix_type
            pix_neighbors_prime = camera_class.pix_neighbors
        except: pass        

        try: pix_area = h.get_pixel_area(tel_id[t])
        except:
            try: pix_area = pix_area_prime
            except: pass

        try: pix_type = h.get_pixel_type(tel_id[t])
        except:
            try: pix_type = pix_type_prime
            except: pix_type = 'unknown'
        
        try: pix_neighbors = h.get_pixel_neighbor(tel_id[t])
        except:
            try: pix_neighbors = pix_neighbors_prime
            except: pass
        
        cam_table_prime.meta = {'TELID': tel_id[t], 'VERSION': version, \
        'PIXTYPE': pix_type, \
        'PIXX_DES': 'x-position of the pixel measured by...'}
        cam_table_prime['PixID'] = pix_id
        cam_table_prime['PixX'] = pix_posX
        cam_table_prime['PixX'].unit = u.m
        cam_table_prime['PixY'] = pix_posY
        cam_table_prime['PixY'].unit = u.m
        cam_table_prime['PixA'] = pix_area
        cam_table_prime['PixA'].unit = u.mm**2
        cam_table_prime['PixNeig'] = pix_neighbors
        
        opt_table_prime.meta = {'TELID': tel_id[t], 'VERSION': version, \
        'MIRN': h.get_mirror_number(tel_id[t]), \
        'MIRA': h.get_mirror_area(tel_id[t])*u.m**2, \
        'FL': h.get_optical_foclen(tel_id[t])*u.m, \
        'MIRA_DES': 'Area of all mirrors'}
        
        camera['CameraTable_Version%s_TelID%i' % (version,tel_id[t])] \
        = cam_table_prime
        optics['OpticsTable_Version%s_TelID%i' % (version,tel_id[t])] \
        = opt_table_prime
        
    print('Astropy tables have been created.')
    h.close_file()
    print("Hessio file has been closed.")
    return(telescope,camera,optics)

def load_fits(filename = '',path = None,version = '',instr_item = ''):
    """
    Function to load data from a fits file
    
    Parameters
    ----------
    filename: string
        name of the file
    path: string
        name of the path, where the file(s) are located
        - when already given as part of the filename, nothing to be entered
        - when file(s) is located in current directory, nothing to be entered
    version: string
        needed if no filename is given and if files with a given version should
        be looked for
    instr_item: string
        item of the instrument (Telescope,Camera or Optics) whose data
        should be loaded, can be either a list, e.g. ('Camera','Telescope') or
        just one string, e.g. 'Camera'
    Return
    ------
    telescope,camera,optics: 3 dictionaries
        all dictionaries contain astropy.table Tables
    """
    print('Fits file will be opened.')
    if filename == '':
        filename = []
        dirListing = os.listdir(path)
        if type(instr_item) is str: instr_item = (instr_item,)
        for item in dirListing:
            for item2 in instr_item:
                if item2 in item:
                    if version in item:
                        filename.append(item)
    else:
        filename = [filename]
    
    telescope = {}
    camera = {}
    optics = {}
    
    for item in filename:
        if 'Telescope' in item:
            telescope[os.path.splitext(item)[0]] = Table.read(item)
        elif 'Camera' in item:
            camera[os.path.splitext(item)[0]] = Table.read(item)
        elif 'Optics' in item:
            optics[os.path.splitext(item)[0]] = Table.read(item)
        else:
            h = 1
            while True:
                try: telescope['%i' % h] = Table.read(item,hdu=h)
                except: break
                h+=1
                if h>100:
                    break
    
    print('Astropy tables have been created.')
    return telescope,camera,optics
    
def load_config(filename):
    """
    Function to load config file data into a table
    
    Parameters
    ----------
    filename: string
        name of the file
    """
    data = read_config_data(filename)
    dirname = os.path.dirname(filename)
    
    version = os.path.splitext(os.path.basename(filename))[0]
    
    telescope = {}
    camera = {}
    optics = {} 
    
    opt_table_prime = Table()
    opt_table_prime.meta = {'VERSION': version}
    try: opt_table_prime['MirClass'] =  data.mirror_class[0]
    except:pass
    try: opt_table_prime['PrimMirParam'] = [data.primary_mirror_parameters]
    except: pass
    try:
        opt_table_prime['PrimRefRad'] = data.primary_ref_radius[0]
        opt_table_prime['PrimRefRad'].unit = u.cm
    except: pass
    try:
        opt_table_prime['PrimDiam'] = data.primary_diameter[0]
        opt_table_prime['PrimDiam'].unit = u.cm
    except: pass
    try:
        opt_table_prime['PrimHoleDiam'] = data.primary_hole_diameter[0]
        opt_table_prime['PrimHoleDiam'].unit = u.cm
    except: pass
    try: opt_table_prime['SndMirParam'] = [data.secondary_mirror_parameters]
    except: pass
    try:
        opt_table_prime['SndRefRad'] = data.secondary_ref_radius[0]
        opt_table_prime['SndRefRad'].unit = u.cm
    except: pass
    try:
        opt_table_prime['SndDiam'] = data.secondary_diameter[0]
        opt_table_prime['SndDiam'].unit = u.cm
    except: pass
    try:
        opt_table_prime['SndHoleDiam'] = data.secondary_hole_diameter[0]
        opt_table_prime['SndHoleDiam'].unit = u.cm
    except: pass
    try: opt_table_prime['FocalSurfParam'] = [data.focal_surface_param]
    except: pass
    try:
        opt_table_prime['FL'] = data.focal_length[0]
        opt_table_prime['FL'].unit = u.cm
    except: pass
    try:
        opt_table_prime['FocalSurfRefRad'] = data.focal_surface_ref_radius[0]
        opt_table_prime['FocalSurfRefRad'].unit = u.cm
    except: pass
    try:
        opt_table_prime['RandmFocalLen'] = data.random_focal_length[0]
        opt_table_prime['RandmFocalLen'].unit = u.cm
    except: pass
    try: opt_table_prime['MirReflRndmAngle'] = data.mirror_reflection_random_angle[0]
    except: pass
    try: opt_table_prime['MirAlignRndmDist'] = data.mirror_align_random_distance[0]
    except: pass
    try: opt_table_prime['MirAlignRndmHoriz'] = data.mirror_align_random_horizontal[0]
    except: pass
    try: opt_table_prime['MirAlignRndmVert'] = data.mirror_align_random_vertical[0]
    except: pass
    try: opt_table_prime['MirRefl'] = data.mirror_reflectivity[0]
    except: pass
    refl_table = Table()
    try:
        mir_refl_filename = \
        dirname+'/'+textwrap.dedent(data.mirror_reflectivity[0])
        mir_refl_version = \
        os.path.splitext(os.path.basename(mir_refl_filename))[0]
        wavel,reflect = \
        np.loadtxt(dirname+'/'+textwrap.dedent(data.mirror_reflectivity[0]),
                                       unpack=True)
        refl_table.meta = {'NAME': 'Mirror reflectivity',
                           'VERSION': mir_refl_version }
        refl_table['Wavelength'] = wavel
        refl_table['Wavelength'].unit = u.nm
        refl_table['Reflectivity'] = reflect
    except: pass
    try: opt_table_prime['MirList'] = data.mirror_list[0]
    except: pass
    try: opt_table_prime['MirOffset'] = data.mirror_offset[0]
    except: pass
    try: opt_table_prime['FocusOffset'] = data.focus_offset[0]
    except: pass
    try: opt_table_prime['TelTrans'] = data.telescope_transmission[0]
    except: pass
    try: opt_table_prime['TelRndmAngle'] = data.telescope_random_angle[0]
    except: pass
    try: opt_table_prime['TelRndmErr'] = data.telescope_random_error[0]
    except: pass
    
    optics['OpticsTable_%s' % version] = opt_table_prime
    optics['MirrorRefelctivity_%s' % mir_refl_version] = refl_table

    return(telescope,camera,optics)

def get_var_from_file(filename):
    """
    Function to load and initialize a module implemented as a Python source
    file called `filename` and to return its module objects.
    
    Parameter
    ---------
    filename: ASCII file
        file in which the module objects are defined
    """
    f = open(filename)
    global data
    data = imp.load_source('data', '', f)
    f.close()
    
def read_config_data(filename):
    """Function to open and load a config file"""
    file = open(filename,'r')
    print("Config file %s will be opened." % filename)
    temp_filename = '%s_temp.txt' % os.path.splitext(os.path.basename(filename))[0]
    print("Temporary file ",temp_filename," created.")
    temp_file = open(temp_filename, 'w')    
    for line in file:
        if 'echo' in line:
            line = line.replace('echo','#')
        if "%" in line:
            line = line.replace('%','#')
        if line.startswith('#'):
            pass
        else:
            if '=' in line:
                index1 = line.index('=')
                if '#' in line:
                    index2 = line.index('#')-1
                else:
                    index2 = len(line)-1
                line = line[:index1+1]+'['+line[index1+1:index2]+']'+'\n'
                for i in range(index1+2,index2+1,1):                    
                    if line[i] == ' ' or line[i] == 'e' or line[i] == 'E':
                        x = True
                        pass
                    elif ord(line[i]) > 57:
                        x = False
                        break
                    else:
                        x = True
                        pass
                if x==False: line = line[:index1+2]+'"'+line[index1+2:index2+1]+'"'+']'+'\n'
            else:
                line = '\n'
        line = textwrap.dedent(line)
        temp_file.write(line) 
    file.close()
    print("Config file has been closed.")
    temp_file.close()
    print("Temporaryly created file has been closed.")
    get_var_from_file(temp_filename)
    os.remove(temp_filename)
    print("Temporaryly created file has been removed.")
    return data


    
def write_fits(filename = '',instr_dict = '', path = '',table_name = ''):
    """
    Function writing data from the astropy.table Table into a fits file
    
    Parameters
    ----------
    filename: string
        if filename is given, the filename is not generated automatically but
        table is written to the given filename (including path!), only possible if one specific
        table is given, i.e. table_name has to be given
    instr_dict: dictionary
        dictionary containing tables with either telescope, camera, or optics
        data
    path: string
        pathname of the directory, where the file(s) should be wirtten to;
        default: current directory
    table_name: string
        name of the table of the directory which should be written into
        the file;
        default: all tables of the directory are wirtten to files
    """
    if table_name == '':
        for key in instr_dict:
            if filename == '':
                instr_dict[key].write('%s%s.fits' % (path,key))
                print('%s%s.fits has been created.' % (path,key))
            else:
                instr_dict[key].write(filename)
                print('%s has been created.' % filename)
    else:
        instr_dict[table_name].write('%s%s.fits' % (path,table_name))
        print('%s%s.fits has been created.' % (path,table_name))