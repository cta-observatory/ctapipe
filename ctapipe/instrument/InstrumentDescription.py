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
    possibility 0:
    telescope,camera,optics = load_fakedata(filename)
    
    possibility 1:
    telescope,camera,optics = load_hessio(filename)
    
    possibility 2:
    telescope,camera,optics = load_fits(filename = '',path = None,
                                        version = '',instr_item = '')
    possibility 3:
    telescope,camera,optics = load_config(filename)
    
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
    
    #Creating 3 dictionaries where the instrument configuration will be stored
    #The dictionaries themselves contain astropy.table.Table objects
    telescope = {}
    camera = {}
    optics = {}  
    
    #--------------------------------------------------------------------------
    #Telescope configuration
    tel_table_prime = Table()
    tel_table_prime.meta = {'VERSION': version}    
    
    try: 
        tel_id = h.get_telescope_ids()
        tel_table_prime['TelID']= tel_id
    except: pass    
    try:
        tel_posX = [h.get_telescope_position(i)[0] for i in tel_id]
        tel_table_prime['TelX'] = tel_posX
        tel_table_prime['TelX'].unit = u.m
        tel_table_prime.meta['TelX_description'] =\
        'x-position of the telescope measured by...'
    except: pass
    try:
        tel_posY = [h.get_telescope_position(i)[1] for i in tel_id]
        tel_table_prime['TelY'] = tel_posY
        tel_table_prime['TelY'].unit = u.m
    except: pass
    try:
        tel_posZ = [h.get_telescope_position(i)[2] for i in tel_id]
        tel_table_prime['TelZ'] = tel_posZ
        tel_table_prime['TelZ'].unit = u.m
    except: pass
    try: tel_table['CameraClass'] = [h.get_camera_class(i) for i in tel_id]
    except: pass
    try:
        tel_table_prime['MirA'] = [h.get_mirror_area(i) for i in tel_id]
        tel_table_prime['MirA'].unit = u.m**2
    except: pass
    try:  tel_table_prime['MirN'] = [h.get_mirror_number(i) for i in tel_id]
    except: pass    
    try: 
        tel_table_prime['FL'] = [h.get_optical_foclen(i) for i in tel_id]
        tel_table_prime['FL'].unit = u.m
    except: pass
    try: tel_table_prime.meta['TelNum'] =  len(tel_posX)
    except: pass
    
    #Beside other tables containimng telescope configuration data, the main
    #telescope table is written into the telescope dictionary.
    telescope['TelescopeTableVersion%s' % version] = tel_table_prime
    
    #--------------------------------------------------------------------------
    #Camera and Optics configuration
    try:    
        for t in range(len(tel_id)):       
            
            cam_table_prime = Table()
            cam_table_prime.meta = {'TELID': tel_id[t], 'VERSION': version}
            opt_table_prime = Table()
            opt_table_prime.meta = {'TELID': tel_id[t], 'VERSION': version}
            
            try:
                pix_posX = h.get_pixel_position(tel_id[t])[0]
                pix_id = np.arange(len(pix_posX))
                cam_table_prime['PixID'] = pix_id                
                cam_table_prime['PixX'] = pix_posX
                cam_table_prime['PixX'].unit = u.m
                cam_table_prime.meta['PixXDescription'] =\
                'x-position of the pixel measured by...'
            except: pass
            try:
                pix_posY = h.get_pixel_position(tel_id[t])[1]
                cam_table_prime['PixY'] = pix_posY
                cam_table_prime['PixY'].unit = u.m
            except: pass
            try:
                camera_class = CD.guess_camera_geometry(pix_posX*u.m,pix_posY*u.m)
                pix_area_prime = camera_class.pix_area
                pix_type_prime = camera_class.pix_type
                pix_neighbors_prime = camera_class.pix_neighbors
            except: pass        
    
            try:
                pix_area = h.get_pixel_area(tel_id[t])
                cam_table_prime['PixA'] = pix_area
                cam_table_prime['PixA'].unit = u.mm**2
            except:
                try:
                    cam_table_prime['PixA'] = pix_area_prime
                    cam_table_prime['PixA'].unit = u.mm**2
                except: pass
            try: pix_type = h.get_pixel_type(tel_id[t])
            except:
                try: pix_type = pix_type_prime
                except: pix_type = 'unknown'
            cam_table_prime.meta['PixType'] = pix_type
            try:
                pix_neighbors = h.get_pixel_neighbor(tel_id[t])
                cam_table_prime['PixNeig'] = pix_neighbors
            except:
                try: cam_table_prime['PixNeig'] = pix_neighbors_prime
                except: pass       
            
            #as long as no mirror IDs are given, use the following:
            opt_table_prime['MirrID'] = [1,2]
            try:
                opt_table_prime.meta['MirNum'] = h.get_mirror_number(tel_id[t])
            except: pass
            try:
                opt_table_prime['MirArea'] = h.get_mirror_area(tel_id[t])
                opt_table_prime['MirArea'].unit = u.m**2
                opt_table_prime.meta['MirAreaDescription'] =\
                'Area of all mirrors'
            except: pass
            try:
                opt_table_prime['OptFocLen'] = h.get_optical_foclen(tel_id[t])
                opt_table_prime['OptFocLen'].unit = u.m
            except: pass
            
            #Beside other tables containing camera and optics configuration
            #data, the main  tables are written into the camera and optics
            #dictionary.
            camera['CameraTable_Version%s_TelID%i' % (version,tel_id[t])] \
            = cam_table_prime
            optics['OpticsTable_Version%s_TelID%i' % (version,tel_id[t])] \
            = opt_table_prime
    except: pass
        
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
    
    #Creating 3 dictionaries where the instrument configuration will be stored
    #The dictionaries themselves contain astropy.table.Table objects
    telescope = {}
    camera = {}
    optics = {}
    
    for item in filename:
        if ('Telescope' or 'Tel') in item:
            telescope[os.path.splitext(item)[0]] = Table.read(item)
        elif ('Camera' or 'Cam') in item:
            camera[os.path.splitext(item)[0]] = Table.read(item)
        elif ('Optics' or 'Opt') in item:
            optics[os.path.splitext(item)[0]] = Table.read(item)
        else:
            #If in the fits file not the same nomenclature is used as in the
            #'write_fits' function or if it contains more than one HDU object,
            #just iterate over all HDUs and write them into one dictionary
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
    Return
    ------
    telescope,camera,optics: 3 dictionaries
        all dictionaries contain astropy.table Tables
    """
    data = read_config_data(filename)
    dirname = os.path.dirname(filename)
    version = os.path.splitext(os.path.basename(filename))[0]
    
    #Creating 3 dictionaries where the instrument configuration will be stored
    #The dictionaries themselves contain astropy.table.Table objects
    telescope = {}
    camera = {}
    optics = {} 
    
    #--------------------------------------------------------------------------
    #Telescope configuration
    tel_table_prime = Table()
    tel_table_prime['TelID']=[1,2]
    tel_table_prime.meta = {'VERSION': version}
    try:
        tel_table_prime['TriggerDelayComp'] = \
        data.trigger_delay_compensation[0]
    except: pass
    try: tel_table_prime['DefaultTrigger'] = data.default_trigger[0]
    except: pass
    try: tel_table_prime['TriggerDiscBins'] = data.disc_bins[0]
    except: pass
    try: tel_table_prime['TriggerDiscStart'] = data.disc_start[0]
    except: pass
    try:
        tel_table_prime['DiscriminatorPulseShape'] = \
        data.discriminator_pulse_shape[0]
    except: pass
    try:
        pulseshape_table = Table()        
        pulseshape_filename = \
        dirname+'/'+textwrap.dedent(data.discriminator_pulse_shape[0]).strip()
        pulseshape_version = \
        os.path.splitext(os.path.basename(pulseshape_filename))[0]
        time,amplitude = \
        np.loadtxt(pulseshape_filename,unpack=True,usecols=(0,1))
        pulseshape_table.meta = {'NAME': 'Discriminator pulse shape',
                           'VERSION': pulseshape_version }
        pulseshape_table['Time'] = time
        pulseshape_table['Time'].unit = u.ns
        pulseshape_table['Amplitude'] = amplitude
        telescope['Tel_DiscriminatorPulseShape'] = pulseshape_table
    except: pass
    try:
        tel_table_prime['DiscriminatorAmplitude'] = \
        data.discriminator_amplitude[0]
        tel_table_prime['DiscriminatorAmplitude'].unit = u.mV
    except:pass
    try: tel_table_prime['NumTriggerPixels'] = trigger_pixels[0]
    except: pass
    try:
        tel_table_prime['DiscriminatorThreshold'] = \
        discriminator_threshold[0]
    except: pass
    try: tel_table_prime['DefaultTrig'] = data.default_trigger[0]
    except: pass
    try:
        tel_table_prime['TelTrigMinTime'] = data.teltrig_min_time[0]
        tel_table_prime['TelTrigMinTime'].unit = u.ns
    except: pass
    try:
        tel_table_prime['TelTrigMinSigSum'] = data.teltrig_min_sigsum[0]
        tel_table_prime['TelTrigMinSigSum'].unit = u.pV * u.s
    except: pass
    try: tel_table_prime['TrigDelayComp'] = data.trigger_delay_compensation[0]
    except: pass
    try:
        tel_table_prime['DiscriminatorVarThreshold'] = \
        discriminator_var_threshold[0]
    except: pass
    try:
        tel_table_prime['SamplingRate'] = data.fadc_MHz[0]
        tel_table_prime['SamplingRate'].unit = u.MHz
    except: pass
    try:
        tel_table_prime['SamplingRate'] = data.fadc_mhz[0]
        tel_table_prime['SamplingRate'].unit = u.MHz
    except: pass
    try: tel_table_prime['FADCPulseShape'] = data.fadc_pulse_shape[0]
    except: pass
    try:
        fadc_pulseshape_table = Table()        
        fadc_pulseshape_filename = \
        dirname+'/'+textwrap.dedent(data.fadc_pulse_shape[0]).strip()
        pulseshape_version = \
        os.path.splitext(os.path.basename(fadc_pulseshape_filename))[0]
        time,amplitude = \
        np.loadtxt(fadc_pulseshape_filename,unpack=True,usecols=(0,1))
        fadc_pulseshape_table.meta = {'NAME': 'FADC pulse shape',
                           'VERSION': pulseshape_version }
        fadc_pulseshape_table['Time'] = time
        fadc_pulseshape_table['Time'].unit = u.ns
        fadc_pulseshape_table['Amplitude'] = amplitude
        telescope['Tel_FADCPulseShape'] = pulseshape_table
    except: pass
    try: tel_table_prime['FADCBins'] = data.fadc_bins[0]
    except: pass
    try: tel_table_prime['FADCSumBins'] = data.fadc_sum_bins[0]
    except: pass
    try: tel_table_prime['FADCSumOffset'] = data.fadc_sum_offset[0]
    except: pass
    try: tel_table_prime['FADCPedestal'] = data.fadc_pedestal[0]
    except: pass
    try: tel_table_prime['FADCAmplitude'] = data.fadc_amplitude[0]
    except: pass
    try: tel_table_prime['FADCNoise'] = data.fadc_noise[0]
    except: pass
    try: tel_table_prime['NumGains'] = data.num_gains[0]
    except: pass
    try: tel_table_prime['FADCLowGainPedestal'] = data.fadc_lg_pedestal[0]
    except: pass
    try: tel_table_prime['FADCLowGainAmplitude'] = data.fadc_lg_amplitude[0]
    except: pass
    try: tel_table_prime['FADCLowGainNoise'] = data.fadc_lg_noise[0]
    except: pass
    try: tel_table_prime['FADCMaxSignal'] = data.fadc_max_signal[0]
    except: pass
    try: tel_table_prime['FADCMaxSum'] = data.fadc_max_sum[0]
    except: pass
    try:
        tel_table_prime['CamBodyDiam'] = data.camera_body_diameter[0]
        tel_table_prime['CamBodyDiam'].unit = u.cm
    except: pass
    try:
        tel_table_prime['CamDepth'] = data.camera_depth[0]
        tel_table_prime['CamDepth'].unit = u.cm
    except: pass
    try: tel_table_prime['CamConfig'] = data.camera_config_file[0]
    except:pass
    try: tel_table_prime['PixNum'] = data.camera_pixels[0]
    except: pass

    #Beside other tables containimng telescope configuration data, the main
    #telescope table is written into the telescope dictionary.
    telescope['TelescopeTable_%s' % version] = tel_table_prime
    
    
    #--------------------------------------------------------------------------
    #Camera configuration
    cam_table_prime = Table()
    cam_table_prime.meta = {'VERSION': version}
    try:
        cam_table_prime.meta['CamBodyDiam'] = \
        data.camera_body_diameter[0]*u.cm
    except: pass
    try: cam_table_prime.meta['CamDepth'] = data.camera_depth[0]*u.cm
    except:pass
    try: cam_table_prime.meta['CamConfig'] = data.camera_config_file[0]
    except:pass
    try:
        cam_config_filename = \
        dirname+'/'+textwrap.dedent(data.camera_config_file[0]).strip()
        pix_posX,pix_posY =\
        np.loadtxt(cam_config_filename,
                   comments = ('#','M','PixType','AnalogSumTrigger','Rotate',\
                   'Trigger','DigitalSumTrigger'),
                   usecols=(3,4),unpack=True)
        board_ID =\
        np.loadtxt(cam_config_filename,
                   comments = ('#','M','PixType','AnalogSumTrigger','Rotate',\
                   'Trigger','DigitalSumTrigger'),
                   usecols=(7,),unpack=True,dtype=str)
        try:            
            pix_ID,module_number,board_number,channel_number,pixel_on = \
            np.loadtxt(cam_config_filename,
                    comments = ('#','M','PixType','AnalogSumTrigger','Rotate',\
                    'Trigger','DigitalSumTrigger'),
                    usecols=(1,2,5,6,8),unpack=True,dtype=int)
            
        except:
            pix_ID,module_number,board_number,channel_number = \
            np.loadtxt(cam_config_filename,
                    comments = ('#','M','PixType','AnalogSumTrigger','Rotate',\
                    'Trigger','DigitalSumTrigger'),
                    usecols=(1,2,5,6),unpack=True,dtype=int)
            pixel_on = [-1]
        cam_table_prime['PixID'] = pix_ID
        cam_table_prime['PixX'] = pix_posX
        cam_table_prime['PixX'].unit = u.cm
        cam_table_prime['PixY'] = pix_posY
        cam_table_prime['PixY'].unit = u.cm
        cam_table_prime['ModuleNum'] = module_number
        cam_table_prime['BoardNum'] = board_number
        cam_table_prime['ChannelNum'] = channel_number
        cam_table_prime['BoardID'] = board_ID
        if pixel_on[0] != -1:
            cam_table_prime['PixelON'] = pixel_on
    except: cam_table_prime['PixID'] = [1,2]
    try:
        cam_config_filename = \
        dirname+'/'+textwrap.dedent(data.camera_config_file[0]).strip()
        try:
            pix_type,pmt_type,cathode_shape_type,visible_cathode_diameter,\
            funnel_shape_type,funnel_diameter,funnel_depth =\
            np.genfromtxt(cam_config_filename,usecols=(1,2,3,4,5,6,7),\
            unpack=True,max_rows=1)
            funnel_efficiency,funnel_wall_reflectivity =\
            np.genfromtxt(cam_config_filename,usecols=(8,9),unpack=True,\
            max_rows=1,dtype=None)
        except:
            pix_type,pmt_type,cathode_shape_type,visible_cathode_diameter,\
            funnel_shape_type,funnel_diameter,funnel_depth=\
            np.genfromtxt(cam_config_filename,usecols=(1,2,3,4,5,6,7),\
            unpack=True,max_rows=1)
            funnel_efficiency =\
            np.genfromtxt(cam_config_filename,usecols=(8,),\
            unpack=True,max_rows=1,dtype=str)
            funnel_wall_reflectivity = [-1]
        cam_table_prime['PixType'] = pix_type
        cam_table_prime['PMTType'] = pmt_type
        cam_table_prime['CathType'] = cathode_shape_type
        cam_table_prime['CathDiam'] = visible_cathode_diameter
        cam_table_prime['CathDiam'].unit = u.cm
        cam_table_prime['FunnelTzpe'] = funnel_shape_type
        cam_table_prime['FunnelDiam'] = funnel_diameter
        cam_table_prime['FunnelDiam'].unit = u.cm
        cam_table_prime['FunnelDepth'] = funnel_depth
        cam_table_prime['FunnelDepth'].unit = u.cm
        cam_table_prime['FunnelEff'] = funnel_efficiency.reshape(1)[0]
        if funnel_wall_reflectivity[0] != -1:
            cam_table_prime['FunnelRefl'] = funnel_wall_reflectivity
    except: pass
    try:
        funnel_eff_table = Table()        
        funnel_eff_filename = \
        dirname+'/'+funnel_efficiency.reshape(1)[0]
        funnel_eff_version = \
        os.path.splitext(os.path.basename(funnel_eff_filename))[0]
        angle,transmission = \
        np.loadtxt(funnel_eff_filename,unpack=True)
        refl_table.meta = {'NAME': 'Funnel efficiency',
                           'VERSION': funnel_eff_version }
        funnel_eff_table['Angle'] = wavel
        funnel_eff_table['Angle'].unit = u.degree
        funnel_eff_table['Transmission'] = transmission
        camera['Cam_FunnelEfficiency'] = funnel_eff_table
    except: pass
    try: cam_table_prime.meta['PixNum'] = data.camera_pixels[0]
    except: pass
    try: cam_table_prime['PhDetEff'] = data.quantum_efficiency[0]
    except:pass
    try:
        quantumeff_table = Table()        
        cam_quantumeff_filename =\
        dirname+'/'+textwrap.dedent(data.quantum_efficiency[0]).strip()
        cam_quantumeff_version = \
        os.path.splitext(os.path.basename(cam_quantumeff_filename))[0]
        wavel,ph_detection_eff = \
        np.loadtxt(cam_quantumeff_filename,unpack=True)
        quantumeff_table.meta = {'NAME': 'Quantum efficiency',
                           'VERSION': cam_quantumeff_version}
        quantumeff_table['Wavelength'] = wavel
        quantumeff_table['Wavelength'].unit = u.nm
        quantumeff_table['DetEfficiency'] = ph_detection_eff
        camera['Cam_PhotonDetectionEfficiency'] = quantumeff_table
    except: pass
    try: cam_table_prime['PESpec'] = data.pm_photoelectron_spectrum[0]
    except: pass
    try:
        pespectrum_table = Table()
        pespectrum_filename =\
        dirname+'/'+textwrap.dedent(data.pm_photoelectron_spectrum[0]).strip()
        pespectrum_version = \
        os.path.splitext(os.path.basename(pespectrum_filename))[0]
        amplitude,probability = \
        np.loadtxt(pespectrum_filename,unpack=True,usecols=(0,1))
        pespectrum_table.meta = {'NAME': 'Photoelectron spectrum',
                           'VERSION': pespectrum_version}
        pespectrum_table['Amplitude [PE]'] = amplitude
        pespectrum_table['Probability'] = probability
        camera['Cam_PhotoelectronSpectrum'] = \
        pespectrum_table
    except: pass
    try: cam_table_prime['TransTimeJitt'] = data.transit_time_jitter[0]
    except: pass
    try: cam_table_prime['GainVar'] = data.gain_variation[0]
    except: pass
    try: cam_table_prime['QEVar'] = data.qe_variation[0]
    except: pass
    try: cam_table_prime['CamFilter'] = data.camera_filter[0]
    except: pass
    try:
        camfilter_table = Table()        
        camfilter_filename = \
        dirname+'/'+textwrap.dedent(data.camera_filter[0]).strip()
        camfilter_version = \
        os.path.splitext(os.path.basename(camfilter_filename))[0]
        wavel,transmission = \
        np.loadtxt(camfilter_filename,unpack=True)
        camfilter_table.meta = {'NAME': 'Camera window transmission',
                           'VERSION': camfilter_version }
        camfilter_table['Wavelength'] = wavel
        camfilter_table['Wavelength'].unit = u.nm
        camfilter_table['Transmission'] = transmission
        camera['CameraWindowTransmission'] = camfilter_table
    except: pass
    
    #some variables which are set for simulation --> no real instrument data
    
    #try: cam_table_prime.meta['MINPH'] = data.min_photons[0]
    #except: pass
    #try: cam_table_prime.meta['MIPE'] = data.min_photoelectrons[0]
    #except: pass
    #try: cam_table_prime.meta['STOREPE'] = data.store_photoelectrons[0]
    #except: pass
    #try: cam_table_prime.meta['NSB'] = data.nightsky_background[0]
    #except: pass
    
    #Beside other tables containimng camer configuration data, the main
    #camera table is written into the camera dictionary.
    camera['CameraTable_%s' % version] = cam_table_prime
    
    
    #--------------------------------------------------------------------------
    #Optics configuration
    opt_table_prime = Table()
    opt_table_prime.meta = {'VERSION': version}
    
    try: opt_table_prime.meta['MirrorList'] = data.mirror_list[0]
    except:pass
    try:
        opt_mirrorlist_filename = \
        dirname+'/'+textwrap.dedent(data.mirror_list[0]).strip()
        file = open(opt_mirrorlist_filename,'r')
        temp_filename = 'mirrorlist_temp.txt'
        temp_file = open(temp_filename, 'w')
        for line in file:
            if '#%' in line:
                line = line.replace('#%',' ')
            temp_file.write(line)
        file.close()
        temp_file.close()        
        mirror_characs = np.loadtxt(temp_filename,unpack=True)
        os.remove(temp_filename)
        try:
            mir_ID = mirror_characs[6]
            opt_table_prime['MirID'] = mir_ID.astype(int)
        except: pass
        try:
            mir_posX = mirror_characs[0]
            opt_table_prime['MirX'] = mir_posX
            opt_table_prime['MirX'].unit = u.cm
        except: pass
        try:
            mir_posY = mirror_characs[1]
            opt_table_prime['MirY'] = mir_posY
            opt_table_prime['MirY'].unit = u.cm
        except: pass
        try:
            mir_posZ = mirror_characs[5]
            opt_table_prime['MirZ'] = mir_posZ
            opt_table_prime['MirZ'].unit = u.cm
        except: pass       
        try:
            mir_diam = mirror_characs[2]
            opt_table_prime['MirDiam'] = mir_diam
            opt_table_prime['MirDiam'].unit = u.cm
        except: pass
        try:
            mir_FL = mirror_characs[3]
            opt_table_prime['MirFL'] = mir_FL
            opt_table_prime['MirFL'].unit = u.cm
        except: pass
        try:
            mir_shape_code = mirror_characs[4]
            opt_table_prime['MirShCode'] = mir_shape_code.astype(int)
        except: pass
        try:
            mir_nx = mirror_characs[7]
            opt_table_prime['n_x'] = mir_nx
        except: pass
        try:
            mir_ny = mirror_characs[8]
            opt_table_prime['n_y'] = mir_ny
        except: pass
        try:
            mir_nz = mirror_characs[9]
            opt_table_prime['n_z'] = mir_nz
        except: pass
        try:
            dist_optax_mirrorcentrum = mirror_characs[10]
            opt_table_prime['DistOptAxMirCent'] = dist_optax_mirrorcentrum
            opt_table_prime['DistOptAxMirCent'].unit = u.cm
        except: pass
    except: pass    
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
    try:
        opt_table_prime['MirReflRndmAngle'] = \
        data.mirror_reflection_random_angle[0]
        opt_table_prime['MirReflRndmAngle'].unit = u.degree
    except: pass
    try:
        opt_table_prime['MirAlignRndmDist'] = \
        data.mirror_align_random_distance[0]
        opt_table_prime['MirAlignRndmDist'].unit = u.cm
    except: pass
    try:
        opt_table_prime['MirAlignRndmHoriz'] = \
        data.mirror_align_random_horizontal
    except: pass
    try:
        opt_table_prime['MirAlignRndmVert'] = \
        data.mirror_align_random_vertical
    except: pass
    try: opt_table_prime['MirRefl'] = data.mirror_reflectivity[0]
    except: pass
    try:
        refl_table = Table()        
        mir_refl_filename = \
        dirname+'/'+textwrap.dedent(data.mirror_reflectivity[0]).strip()
        mir_refl_version = \
        os.path.splitext(os.path.basename(mir_refl_filename))[0]
        wavel,reflect = \
        np.loadtxt(mir_refl_filename,unpack=True)
        refl_table.meta = {'NAME': 'Mirror reflectivity',
                           'VERSION': mir_refl_version }
        refl_table['Wavelength'] = wavel
        refl_table['Wavelength'].unit = u.nm
        refl_table['Reflectivity'] = reflect
        optics['Opt_MirrorRefelctivity'] = refl_table
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

    #Beside other tables containimng optics configuration data, the main
    #optics table is written into the optics dictionary.
    optics['OpticsTable_%s' % version] = opt_table_prime  
    
    print('Astropy tables have been created.')
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
    temp_filename =\
    '%s_temp.txt' % os.path.splitext(os.path.basename(filename))[0]
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
                if x==False:
                    line =\
                    line[:index1+2]+'"'+line[index1+2:index2+1]+'"'+']'+'\n'
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


    
def write_fits(filename = '', path = '',instr_dict = '',table_name = '',\
    overwrite=False):
    """
    Function writing data from the astropy.table Table into a fits file
    Every table is written in an own file; fits files with multiple HDU objects
    cannot be created!
    
    Parameters
    ----------
    filename: string
        if filename is given, the table is written to the given filename
        (path must be part of the filename!), only possible if one specific
        table is given, i.e. table_name and instr_dict have to be given.
        If no filename is given, it is generated automatically using the names
        of the given tables amd instr_dict
    path: string
        pathname of the directory, where the file(s) should be wirtten to;
        default: current directory
    instr_dict: dictionary
        dictionary containing tables with either telescope, camera, or optics
        data; has to specified in every case
    table_name: string
        name of the table of the dictionary which should be written into
        the file; if table_name is given, instr_dict must also be specified
        default: all tables of the dictionary are wirtten to files
    """
    if instr_dict == '':
        return(print('instrument dictionary (instr_dict) must be specified.'))
    if table_name == '':
        for key in instr_dict:
            if filename == '':
                write_name = '%s%s.fits' % (path,key)
                instr_dict[key].write(write_name,overwrite)
            else:
                write_name = filename
                instr_dict[key].write(filename,overwrite)
    else:
        if filename == '':
            write_name = '%s%s.fits' % (path,table_name)
            instr_dict[table_name].write(write_name,overwrite)
        else:
            write_name = filename
            instr_dict[table_name].write(filename,overwrite)
    
    return(print('%s has been created' % write_name))