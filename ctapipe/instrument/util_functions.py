import pyhessio as h
import numpy as np
from astropy.io import fits
from astropy.table import Table,join
import random
import astropy.units as u

__all__ = ['get_file_type','load_hessio','load_fits','close_fits',
           'load_fakedata']
           

def get_file_type(filename):
    """
    Returns a string with the type of the given file (guessed from the
    extension). The '.gz' or '.bz2' compression extensions are
    ignored.

    >>> get_file_type('myfile.fits.gz')
    'fits'

    """
    if  ('fit' or 'FITS' or 'FIT') in filename or filename == 'fake_data':
        return 'fits'
    elif 'cfg' in filename:
        return 'ascii'
    else:
        return 'hessio'
    
def load_hessio(filename):
    """
    Function to open and load a hessio file
    
    Parameters
    ----------
    filename: string
        name of the file
    """
    h.file_open(filename)
    print("Hessio file %s has been opened" % filename)
    next(h.move_to_next_event())
    tel_id = h.get_telescope_ids()
    
    tel_table = Table()
    
    tel_table['TelID']= tel_id
    
    tel_posX = [h.get_telescope_position(i)[0] for i in tel_id]
    tel_posY = [h.get_telescope_position(i)[1] for i in tel_id]
    tel_posZ = [h.get_telescope_position(i)[2] for i in tel_id]
    tel_table['TelX'] = tel_posX
    tel_table['TelX'].unit = u.m
    tel_table['TelY'] = tel_posY
    tel_table['TelY'].unit = u.m
    tel_table['TelZ'] = tel_posZ
    tel_table['TelZ'].unit = u.m
    #tel_table['CameraClass'] = [h.get_camera_class(i) for i in tel_id]
    tel_table['MirrorArea'] = [h.get_mirror_area(i) for i in tel_id]
    tel_table['MirrorArea'].unit = u.m**2
    tel_table['NMirrors'] = [h.get_mirror_number(i) for i in tel_id]
    tel_table['FL'] = [h.get_optical_foclen(i) for i in tel_id]
    tel_table['FL'].unit = u.m
    
    
    for t in range(len(tel_id)):       
        
        table = Table()
        pix_posX = h.get_pixel_position(tel_id[t])[0]
        pix_posY = h.get_pixel_position(tel_id[t])[1]       
        pix_id = np.arange(len(pix_posX))
        pix_area = h.get_pixel_area(tel_id[t])
         
        table['TelID'] = [tel_id[t] for i in range(len(pix_posX))]
        table['PixelID'] = pix_id
        table['PixX'] = pix_posX
        table['PixX'].unit = u.m
        table['PixY'] = pix_posY
        table['PixY'].unit = u.m
        table['PixArea'] = pix_area
        table['PixArea'].unit = u.mm**2
        
        if t == 0:
            cam_table = table
        else:
            cam_table = join(cam_table,table,join_type='outer')  
    
    #for t in range(len(tel_id)):
    #    table = Table()
    #    if t == 0:
    #        opt_table = table
    #    else:
    #        opt_table = join(opt_table,table,join_type='outer')
    opt_table = Table()
    print('Astropy tables have been created.')
    
    #to use this, one has to go through every event of the run...
    #n_channel = h.get_num_channel(tel_id)
    #ld.channel_num = n_channel
    #for chan in range(n_channel):
    #    ld.adc_samples.append(h.get_adc_sample(tel_id,chan).tolist())
    
    h.close_file()
    print("Hessio file has been closed.")
    return [tel_table,cam_table,opt_table]

def load_fits(filename):
    """
    Function to open and load a fits file
    
    Parameters
    ----------
    filename: string
        name of the file
    """
    table = []
    if 'fake_data' in filename:
        hdulist = load_fakedata()
        filename = hdulist
    else:
        hdulist = fits.open(filename)
        
    print("Fits file %s has been opened" % filename)
    
    table = []
    for i in range(len(hdulist)):
        try:
            table.append(Table.read(filename,hdu=i))
        except:
            pass
        
    return table
    
def close_fits(hdulist):
    """
    Function to close a fits file
    
    Parameter
    ---------
    hdulist: HDUList
        HDUList object of the fits file
    """    
    hdulist.close()
    print("Fits file has been closed.")   

def load_fakedata():
    """
    Function writing faked date into an astropy.io.fits.hdu.table.BinTableHDU
    """
    tel_num = 10
    tel_id = [random.randrange(1,124,1) for x in range(0,tel_num)]
    tel_posX = [random.uniform(1,100) for x in range(0,tel_num)]
    tel_posY = [random.uniform(1,100) for x in range(0,tel_num)]
    tel_posZ = [random.uniform(1,100) for x in range(0,tel_num)]
    l0id = [i for i in range(0,tel_num)]
    
    mirror_area = [random.uniform(1,100) for x in range(0,5)]
    mirror_number = [random.randrange(1,124,1) for x in range(0,tel_num)]
    
    col0 = fits.Column(name='L0ID',format='I',array=l0id)
    col1 = fits.Column(name='TelID',format='I',array=tel_id)
    col2 = fits.Column(name='TelX',format='D',array=tel_posX)
    col3 = fits.Column(name='TelY',format='D',array=tel_posY)
    col4 = fits.Column(name='TelZ',format='D',array=tel_posZ)
    col5 = fits.Column(name='NMirrors',format='I',array=mirror_number)
    col6 = fits.Column(name='MirrorArea',format='D',array=mirror_area)
    
    cols1 = fits.ColDefs([col0,col1,col2,col3,col4,col5,col6])
    tbhdu1 = fits.BinTableHDU.from_columns(cols1)
    
    pixel_num = 1140    
    l1id = [i for i in range(0,tel_num*pixel_num)]    
    pixel_posX = [random.uniform(1,100) for x in range(0,tel_num*pixel_num)]
    pixel_posY = [random.uniform(1,100) for x in range(0,tel_num*pixel_num)]
    l0id_prime = np.array([[l0id[i] for k in range(0,pixel_num)] for i in range(0,tel_num)])
    l0id_prime = l0id_prime.flatten()
    pixel_id = np.array([[i for i in range(1,pixel_num+1)] for j in range(0,tel_num)])
    pixel_id = pixel_id.flatten()
    
    col01 = fits.Column(name='L1ID',format='I',array=l1id)
    col11 = fits.Column(name='L0ID',format='I',array=l0id_prime)
    col21 = fits.Column(name='PixelID',format='I',array=pixel_id)
    col31 = fits.Column(name='XTubeMM',format='D',array=pixel_posX)
    col41 = fits.Column(name='YTubeMM',format='D',array=pixel_posY)
    
    cols2 = fits.ColDefs([col01,col11,col21,col31,col41])
    tbhdu2 = fits.BinTableHDU.from_columns(cols2)
    
    hdulist = fits.HDUList([tbhdu1,tbhdu2])
    
    return hdulist
    